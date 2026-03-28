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


def write_if_changed(filepath: str, content: str) -> bool:
    """Write content to file only if it differs from the existing file.

    Preserves the file's mtime when content is identical, which prevents
    downstream timestamp-based staleness checks from triggering unnecessary
    reassembly and repackaging.

    Returns True if the file was written, False if unchanged.
    """
    if os.path.isfile(filepath):
        try:
            with open(filepath, "r") as f:
                if f.read() == content:
                    return False
        except (OSError, UnicodeDecodeError):
            pass  # Fall through to write.
    with open(filepath, "w") as f:
        f.write(content)
    return True

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

# Mnemonic substrings for conversion types handled via LLVM IR (not .conv.).
# .conv. (bf16/fp32) is excluded because its intrinsics have different signatures.
_CONVERSION_IR_SUFFIXES = (".ups.", ".srs.", ".pack.", ".unpack.", ".conv.")

# Mapping from base mnemonic (stripped of .2d/.3d) to LLVM intrinsic info.
# Each entry describes:
#   intrinsic: the Peano intrinsic name (without @llvm.aie2. prefix)
#   in_type:   LLVM IR type of the input vector/accumulator
#   out_type:  LLVM IR type of the output vector/accumulator
#   in_bytes:  byte size of in_type
#   out_bytes: byte size of out_type
#   sign:      1 for signed, 0 for unsigned (UPS/SRS only)
#
# UPS intrinsics: load a vector, upshift into an accumulator.
# SRS intrinsics: shift-round-saturate an accumulator into a vector.
# PACK intrinsics: pack wider elements into narrower.
# UNPACK intrinsics: unpack narrower elements into wider.
CONVERSION_INTRINSICS: dict[str, dict] = {
    # --- UPS (8): vector -> accumulator ---
    "vlda.ups.s32.s16": {
        "intrinsic": "acc32.v16.I256.ups",
        "in_type": "<16 x i16>", "out_type": "<8 x i64>",
        "in_bytes": 32, "out_bytes": 64, "sign": 1,
    },
    "vlda.ups.s32.d16": {
        "intrinsic": "acc32.v16.I256.ups",
        "in_type": "<16 x i16>", "out_type": "<8 x i64>",
        "in_bytes": 32, "out_bytes": 64, "sign": 0,
    },
    "vlda.ups.s32.s8": {
        "intrinsic": "acc32.v32.I256.ups",
        "in_type": "<32 x i8>", "out_type": "<16 x i64>",
        "in_bytes": 32, "out_bytes": 128, "sign": 1,
    },
    "vlda.ups.s32.d8": {
        "intrinsic": "acc32.v32.I256.ups",
        "in_type": "<32 x i8>", "out_type": "<16 x i64>",
        "in_bytes": 32, "out_bytes": 128, "sign": 0,
    },
    "vlda.ups.s64.s32": {
        "intrinsic": "acc64.v8.I256.ups",
        "in_type": "<8 x i32>", "out_type": "<8 x i64>",
        "in_bytes": 32, "out_bytes": 64, "sign": 1,
    },
    "vlda.ups.s64.d32": {
        "intrinsic": "acc64.v8.I256.ups",
        "in_type": "<8 x i32>", "out_type": "<8 x i64>",
        "in_bytes": 32, "out_bytes": 64, "sign": 0,
    },
    "vlda.ups.s64.s16": {
        "intrinsic": "acc64.v16.I256.ups",
        "in_type": "<16 x i16>", "out_type": "<16 x i64>",
        "in_bytes": 32, "out_bytes": 128, "sign": 1,
    },
    "vlda.ups.s64.d16": {
        "intrinsic": "acc64.v16.I256.ups",
        "in_type": "<16 x i16>", "out_type": "<16 x i64>",
        "in_bytes": 32, "out_bytes": 128, "sign": 0,
    },
    # --- SRS (8): accumulator -> vector ---
    "vst.srs.s16.s32": {
        "intrinsic": "I256.v16.acc32.srs",
        "in_type": "<8 x i64>", "out_type": "<16 x i16>",
        "in_bytes": 64, "out_bytes": 32, "sign": 1,
    },
    "vst.srs.d16.s32": {
        "intrinsic": "I256.v16.acc32.srs",
        "in_type": "<8 x i64>", "out_type": "<16 x i16>",
        "in_bytes": 64, "out_bytes": 32, "sign": 0,
    },
    "vst.srs.s8.s32": {
        "intrinsic": "I256.v32.acc32.srs",
        "in_type": "<16 x i64>", "out_type": "<32 x i8>",
        "in_bytes": 128, "out_bytes": 32, "sign": 1,
    },
    "vst.srs.d8.s32": {
        "intrinsic": "I256.v32.acc32.srs",
        "in_type": "<16 x i64>", "out_type": "<32 x i8>",
        "in_bytes": 128, "out_bytes": 32, "sign": 0,
    },
    "vst.srs.s32.s64": {
        "intrinsic": "I256.v8.acc64.srs",
        "in_type": "<8 x i64>", "out_type": "<8 x i32>",
        "in_bytes": 64, "out_bytes": 32, "sign": 1,
    },
    "vst.srs.d32.s64": {
        "intrinsic": "I256.v8.acc64.srs",
        "in_type": "<8 x i64>", "out_type": "<8 x i32>",
        "in_bytes": 64, "out_bytes": 32, "sign": 0,
    },
    "vst.srs.s16.s64": {
        "intrinsic": "I256.v16.acc64.srs",
        "in_type": "<16 x i64>", "out_type": "<16 x i16>",
        "in_bytes": 128, "out_bytes": 32, "sign": 1,
    },
    "vst.srs.d16.s64": {
        "intrinsic": "I256.v16.acc64.srs",
        "in_type": "<16 x i64>", "out_type": "<16 x i16>",
        "in_bytes": 128, "out_bytes": 32, "sign": 0,
    },
    # --- PACK (4): wider -> narrower (fuses with store) ---
    # Pack intrinsics take <32 x i16> input + i32 sign flag, return <32 x i8>.
    # The sign flag (0=unsigned, 1=signed) selects d/s prefix.
    # llc fuses pack+store into vst.pack.* instructions.
    "vst.pack.s4.s8": {
        "intrinsic": "pack.I4.I8",
        "in_type": "<32 x i16>", "out_type": "<32 x i8>",
        "in_bytes": 64, "out_bytes": 32, "sign": 1,
    },
    "vst.pack.d4.d8": {
        "intrinsic": "pack.I4.I8",
        "in_type": "<32 x i16>", "out_type": "<32 x i8>",
        "in_bytes": 64, "out_bytes": 32, "sign": 0,
    },
    "vst.pack.s8.s16": {
        "intrinsic": "pack.I8.I16",
        "in_type": "<32 x i16>", "out_type": "<32 x i8>",
        "in_bytes": 64, "out_bytes": 32, "sign": 1,
    },
    "vst.pack.d8.d16": {
        "intrinsic": "pack.I8.I16",
        "in_type": "<32 x i16>", "out_type": "<32 x i8>",
        "in_bytes": 64, "out_bytes": 32, "sign": 0,
    },
    # --- UNPACK (4): narrower -> wider (load + expand) ---
    # Unpack intrinsics take <32 x i8> input + i32 sign flag, return <32 x i16>.
    # llc may or may not fuse with load.
    "vldb.unpack.s8.s4": {
        "intrinsic": "unpack.I8.I4",
        "in_type": "<32 x i8>", "out_type": "<32 x i16>",
        "in_bytes": 32, "out_bytes": 64, "sign": 1,
    },
    "vldb.unpack.d8.d4": {
        "intrinsic": "unpack.I8.I4",
        "in_type": "<32 x i8>", "out_type": "<32 x i16>",
        "in_bytes": 32, "out_bytes": 64, "sign": 0,
    },
    "vldb.unpack.s16.s8": {
        "intrinsic": "unpack.I16.I8",
        "in_type": "<32 x i8>", "out_type": "<32 x i16>",
        "in_bytes": 32, "out_bytes": 64, "sign": 1,
    },
    "vldb.unpack.d16.d8": {
        "intrinsic": "unpack.I16.I8",
        "in_type": "<32 x i8>", "out_type": "<32 x i16>",
        "in_bytes": 32, "out_bytes": 64, "sign": 0,
    },
    # --- BF16/FP32 conversions (2): bf16 <-> fp32 via accumulator ---
    # These fuse with load/store into vlda.conv.fp32.bf16 / vst.conv.bf16.fp32.
    "vlda.conv.fp32.bf16": {
        "intrinsic": "v16bf16.to.v16accfloat",
        "in_type": "<16 x bfloat>", "out_type": "<8 x i64>",
        "in_bytes": 32, "out_bytes": 64,
    },
    "vst.conv.bf16.fp32": {
        "intrinsic": "v16accfloat.to.v16bf16",
        "in_type": "<8 x i64>", "out_type": "<16 x bfloat>",
        "in_bytes": 64, "out_bytes": 32,
    },
}

# Mnemonics that interact with the stream switch -- need live streams.
STREAM_MNEMONICS = frozenset({
    "mov.nb", "mov.nb.tlast", "mov.tlast",
    "mov.ph", "mov.ph.nb", "mov.ph.nb.tlast", "mov.ph.tlast",
    "mov.cph", "mov.cph.nb", "mov.cph.nb.tlast", "mov.cph.tlast",
})

# Register kinds we know how to load/store.
KNOWN_REGISTER_KINDS = frozenset({
    "scalar", "pointer", "vector256", "vector512", "accumulator",
    "control", "modifier_m", "modifier_dj",
    # quad: 128-byte (1024-bit) quadword vector registers q0-q3.
    # These appear in DMV_Q load/store instructions as "accumulator" bw=2
    # in the ISA JSON (exporter misclassification); we promote them here.
    "quad",
    # wide_y: 1024-bit wide vector registers y2-y5 (eYs class).
    # Used as the wide source operand in sparse vector multiply.
    # Each y register is a pair of x registers: y2={x4,x5}, y3={x6,x7}, etc.
    "wide_y",
    # sparse_qx: 640-bit sparse vector+mask composite registers qx0-qx3.
    # Used as the sparse data source in vmac/vmul/vneg/vsub family.
    # Each qx register combines an x register (512-bit vector) and a q
    # register (128-bit mask): qx0={x0,q0}, qx1={x1,q1}, etc.
    "sparse_qx",
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
#   ERS4 (bw=2, 4 entries: r16-r19)
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
_CM_CLASS_OPERAND_NAMES = frozenset({"dst", "src", "acc1", "acc2", "cdst", "csrc"})


def _resolve_unknown_operand(op: dict) -> dict:
    """Attempt to resolve an unknown-typed operand to a concrete register class.

    Handles three exporter limitations:

    1. cm-class (1024-bit full accumulator, eCM/mCMm TableGen class):
       operand_type="unknown", register_kind=None, bit_width=4 with names in
       _CM_CLASS_OPERAND_NAMES -> register_kind="accumulator".

    2. eYs (1024-bit wide vector, y2-y5 for sparse multiply):
       operand_type="unknown", bit_width=2, name="ys1" -> register_kind="wide_y".
       Source: AIE2GenRegisterInfo.td eYs = {y2, y3, y4, y5}.

    All other unknown operands are returned unchanged.
    """
    if op.get("operand_type") != "unknown":
        return op

    name = op.get("name", "")
    bw = op.get("bit_width")

    # cm-class: bw=4, no register_kind, known cm-class name.
    if bw == 4 and not op.get("register_kind") and name in _CM_CLASS_OPERAND_NAMES:
        resolved = dict(op)
        resolved["operand_type"] = "register"
        resolved["register_kind"] = "accumulator"
        return resolved

    # eYs wide vector: bw=2, name=ys1.
    # These appear in sparse vector multiply (vmac, vmul, etc.) as the
    # wide source operand.  The 2-bit field encodes y2-y5 (4 entries).
    if bw == 2 and name == "ys1":
        resolved = dict(op)
        resolved["operand_type"] = "register"
        resolved["register_kind"] = "wide_y"
        return resolved

    return op


def _resolve_sparse_qx(op: dict) -> dict:
    """Resolve accumulator bw=2 operands named qxs2 to sparse_qx kind.

    The ISA exporter classifies mQQXw (640-bit sparse vector+mask composite,
    qx0-qx3) as accumulator bw=2.  This conflicts with mQQa (128-byte quad
    registers q0-q3) which also appears as accumulator bw=2 in load contexts.

    Disambiguation: the operand name "qxs2" is unique to sparse multiply.
    Source: AIE2GenFixupInstrInfo.td vmac_*_core_sparse_wide encodings.
    """
    if op.get("register_kind") != "accumulator":
        return op
    if op.get("bit_width") != 2:
        return op
    if op.get("name") != "qxs2":
        return op
    resolved = dict(op)
    resolved["register_kind"] = "sparse_qx"
    return resolved


def _resolve_operand(op: dict) -> dict:
    """Apply all operand resolution heuristics in sequence."""
    op = _resolve_unknown_operand(op)
    op = _resolve_sparse_qx(op)
    return op


def _effective_kind(op: dict) -> str:
    """Return the effective register kind for an operand.

    For plain register operands, returns register_kind directly.
    For composite_register operands whose kind is in TESTABLE_COMPOSITE_KINDS,
    returns the mapped concrete kind (e.g., "ShflDst" -> "vector512").
    For unknown operands that match the cm-class heuristic (see
    _resolve_unknown_operand), returns "accumulator".
    Returns "" for all other operand types.
    """
    # Apply all operand resolution heuristics before the type dispatch below.
    op = _resolve_operand(op)
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
    # wide_y: 128-byte (1024-bit) wide vector registers y2-y5.
    # Each y is a pair of x registers (2 x 64 bytes).
    "wide_y": 128,
    # sparse_qx: 80-byte (640-bit) sparse vector+mask composite qx0-qx3.
    # 64-byte x component (512-bit vector) + 16-byte q mask (128-bit).
    "sparse_qx": 80,
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
                            in_offset: int, out_offset: int,
                            code_iw_offset: int = 0) -> str:
        """Generate assembly for one test point.

        Args:
            code_iw_offset: Instruction word index of the first instruction
                in this test point (counted from the function entry).  Used
                by BranchStrategy to compute absolute branch targets.
        """
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

    # 4. NOPs and side-effect-only -- skip.
    if mnemonic in SIDE_EFFECT_MNEMONICS:
        return ("skipped", "no output (nop/side-effect)")

    # 4b. Stream switch status reads and stream instructions are now handled
    # by StreamStrategy in the strategy chain.
    asm = instr.get("asm_string", "")
    # 4b2. doTlast_reg variants have $tlast in asm_string that llvm-mc
    # doesn't support.  StreamStrategy rejects them, but they can slip
    # through to ComputeStrategy.  Catch them here as a safety net.
    if "$tlast" in asm:
        return ("skipped", "doTlast_reg variant (unsupported by llvm-mc)")
    # 4c. Cascade write stalls without a downstream consumer tile
    # (confirmed on real NPU: batch_62 hang, 2026-03-24).  Cascade reads
    # stall without an upstream producer.  Both need multi-tile harness.
    if "MCD" in asm:
        return ("skipped", "cascade write (stalls without downstream consumer)")
    # 4d. VEXTRACT/VEXTBCST with immediate index: can't mask at runtime
    # to avoid index-0 silicon errata (see 2026-03-24 investigation).
    # Register-index variants are handled by post-load masking below.
    iname = instr.get("name", "")
    if iname.startswith(("VEXTRACT", "VEXTBCST")) and "ExtractIdxImm" in iname:
        return ("skipped", "immediate-index extract (index-0 errata, can't mask)")

    operands = instr.get("operands", [])

    # 6. No operands at all -> no output.
    if not operands:
        return ("skipped", "no operands")

    # Resolve misclassified operands before all subsequent checks.
    # _resolve_operand handles:
    #   - cm-class unknowns (bw=4, name in dst/src/acc1/acc2/cdst/csrc)
    #   - eYs wide vectors (bw=2, name=ys1) -> wide_y
    #   - mQQXw sparse composites (accumulator bw=2, name=qxs2) -> sparse_qx
    operands = [_resolve_operand(op) for op in operands]

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

    # 10. Skip instructions with unresolved accumulator bw=2 operands.
    #   After _resolve_operand, qxs2 (sparse) becomes "sparse_qx" and ys1
    #   (wide vector) becomes "wide_y", so they won't match here.  Only
    #   genuinely unresolved bw=2 accumulators (e.g. non-qxs2 names in load
    #   contexts) are skipped.
    #
    #   Accumulator bw=6 (256-bit quarters: amll/amlh/amhl/amhh) are now
    #   testable -- they can be loaded/stored via their parent half-accumulator
    #   (bml/bmh) using the existing vups/vsrs infrastructure.
    for op in operands:
        if op.get("register_kind") == "accumulator":
            bw = op.get("bit_width", 0)
            if bw == 2:
                return ("skipped", "composite sparse register (bw=2)")

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

    # Resolve misclassified operands so that dst/src operands with the
    # cm-class signature (and ys1/qxs2 sparse) are treated correctly below.
    operands = [_resolve_operand(op) for op in operands]

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
    # 64-bit element variants (VEXTRACT_D64, VEXTRACT_S64, etc.)
    if name.endswith(("_64", "D64", "S64")):
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

    # ERS4: narrow scalar subclass r16-r19, used as index in VEXTRACT.
    # These are the only 4 registers in this class (2-bit encoding).
    # Source: AIE2GenRegisterInfo.td line 456: def eRS4 = (add r16, r17, r18, r19)
    if kind == "ERS4":
        return ["r16", "r17", "r18", "r19"]

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

    # wide_y: 1024-bit wide vector registers y2-y5 (eYs class).
    # Source: AIE2GenRegisterInfo.td -- y2={x4,x5}, y3={x6,x7}, y4={x8,x9}, y5={x10,x11}.
    if kind == "wide_y":
        return ["y2", "y3", "y4", "y5"]

    # sparse_qx: 640-bit sparse vector+mask composite registers qx0-qx3 (mQQXw class).
    # Source: AIE2GenRegisterInfo.td -- qx0={x0,q0}, qx1={x1,q1}, etc.
    if kind == "sparse_qx":
        return ["qx0", "qx1", "qx2", "qx3"]

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
    if kind == "wide_y":
        # 1024-bit wide vector registers y2-y5 (eYs class).
        # Each y register is a pair of x registers: y2={x4,x5}, y3={x6,x7}, etc.
        # Load by filling all 4 vector256 halves: wl{lo}, wh{lo}, wl{hi}, wh{hi}.
        # y2=x4:x5 -> wl4,wh4,wl5,wh5.  y3=x6:x7 -> wl6,wh6,wl7,wh7.
        y_idx = int(reg_name[1:])  # "y2" -> 2
        lo_x = y_idx * 2       # y2 -> x4
        hi_x = lo_x + 1        # y2 -> x5
        return (
            [f"  // load wide_y {reg_name}: x{lo_x} + x{hi_x} via 4x vlda"]
            + _vector_load(f"wl{lo_x}", ptr, offset)
            + _vector_load(f"wh{lo_x}", ptr, offset + 32)
            + _vector_load(f"wl{hi_x}", ptr, offset + 64)
            + _vector_load(f"wh{hi_x}", ptr, offset + 96)
        )
    if kind == "sparse_qx":
        # 640-bit sparse vector+mask composite registers qx0-qx3 (mQQXw class).
        # Each qx register combines an x register (512-bit) and a q mask (128-bit).
        # Load both: x component via vlda (64 bytes), q mask via lda (16 bytes).
        # qx0={x0,q0} -> load wl0,wh0,q0.  qx1={x1,q1} -> load wl1,wh1,q1.
        qx_idx = int(reg_name[2:])  # "qx0" -> 0
        # q mask starts at offset+64 (after the 64-byte x component).
        # lda qN has scale=16, so immediate range is [-512, 496].
        q_offset = offset + 64
        MAX_QUAD_OFFSET = 496
        if 0 <= q_offset <= MAX_QUAD_OFFSET:
            q_load = [f"  lda q{qx_idx}, [{ptr}, #{q_offset}]"]
        else:
            q_load = _padda_sequence("p6", ptr, q_offset) + [
                f"  lda q{qx_idx}, [p6, #0]",
            ]
        return (
            [f"  // load sparse_qx {reg_name}: x{qx_idx} + q{qx_idx} mask"]
            + _vector_load(f"wl{qx_idx}", ptr, offset)
            + _vector_load(f"wh{qx_idx}", ptr, offset + 32)
            + q_load
        )
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
        if name.startswith("_"):
            continue  # Skip internal metadata keys (_combo_idx, etc.)
        value = regs[name]
        # Immediate values (numeric) need # prefix for llvm-mc syntax.
        if value.lstrip("-").isdigit():
            value = f"#{value}"
        result = result.replace(f"${name}", value)

    # Replace any remaining $name tokens with known fixed registers.
    for name, fixed_reg in FIXED_REGISTER_MAP.items():
        result = result.replace(f"${name}", fixed_reg)

    # Tied-destination: $dst appears in the asm_string but has no matching
    # operand in the operands list (it's an alias for acc1, the in-place
    # accumulate target).  Replace with acc1's value if present.
    if "$dst" in result and "dst" not in regs:
        tied_val = regs.get("acc1", regs.get("acc2", ""))
        if tied_val:
            result = result.replace("$dst", tied_val)

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
        align = 32 if kind in ("vector256", "vector512", "accumulator",
                                "wide_y", "sparse_qx", "quad") else 4
        cur_in_offset = _align(cur_in_offset, align)
        reg = regs.get(op_name, "r0")
        load_lines = _load_instruction(reg, kind, "p0", cur_in_offset)
        lines.extend(load_lines)
        cur_in_offset += _operand_size(op, name)

    # VINSERT: load implicit index register r29 from input buffer.
    # r29 is a fixed implicit register (not a decoded operand) -- the harness
    # must load it explicitly since it won't appear in the operand list.
    if name.startswith("VINSERT"):
        align = 4
        cur_in_offset = _align(cur_in_offset, align)
        lines.extend(_scalar_load("r29", "p0", cur_in_offset))
        cur_in_offset += 4

    # NOP sled before masking/instruction: must cover load pipeline latency.
    # AIE2 lda/vlda latency is 7 cycles (from AIE2Schedule.td).
    # IMPORTANT: the sled must come BEFORE the index masking below, because
    # the index register is loaded via `lda` and AIE2 has no scoreboard --
    # masking before the load pipeline drains reads stale register values.
    lines.extend(_nop_sled(LOAD_LATENCY))

    # Mask VEXTRACT/VINSERT/VEXTBCST index registers to valid element range.
    # Unbounded indices from random test data cause hardware hangs (TDR).
    if name.startswith(("VEXTRACT", "VINSERT", "VEXTBCST")):
        # Determine max valid index from element size in instruction name.
        # Names like VEXTBCST_64_mRm, VEXTRACT_D32, etc. -- match the
        # size component anywhere in the name, not just at the end.
        if "_64" in name or "_D64" in name or "_S64" in name:
            max_idx = 7   # 8 x 64-bit elements in 512-bit vector
        elif "_32" in name or "_D32" in name or "_S32" in name:
            max_idx = 15  # 16 x 32-bit elements
        elif "_16" in name or "_D16" in name or "_S16" in name:
            max_idx = 31  # 32 x 16-bit elements
        else:
            max_idx = 63  # 64 x 8-bit elements
        # VINSERT: index is in implicit r29, not a decoded operand.
        if name.startswith("VINSERT"):
            lines.append(f"  mov r14, #{max_idx}")
            lines.append(f"  and r29, r29, r14")
        else:
            # Find the index register: operand named "idx" with ERS4 or scalar kind.
            # VEXTRACT uses ERS4; VEXTBCST uses scalar. Both need masking.
            idx_op = op_by_name.get("idx", {})
            if idx_op.get("register_kind") in ("ERS4", "scalar"):
                idx_reg = regs.get("idx", "r16")
                # AIE2 AND is register-only; load mask into r14 first.
                lines.append(f"  mov r14, #{max_idx}")
                lines.append(f"  and {idx_reg}, {idx_reg}, r14")
                # VEXTRACT/VEXTBCST index-0 errata: element index 0 after
                # vlda permanently stalls the AIE2 core (Phoenix NPU, confirmed
                # 2026-03-24).  OR with 1 ensures the index is always >= 1.
                # This sacrifices even-index coverage but avoids the hang.
                # See docs/investigations/2026-03-24-vextract-index0-hang.md.
                if name.startswith(("VEXTRACT", "VEXTBCST")):
                    lines.append(f"  mov r14, #1")
                    lines.append(f"  or {idx_reg}, {idx_reg}, r14")

    # Mask VSHIFT/VSHIFT_ALIGN shift amount to valid byte range (1-63).
    # The shift operand is a scalar register loaded from random 32-bit data.
    # Out-of-range values (>=64) produce all-zero output on real hardware.
    # OR with 1 avoids shift=0 which triggers a stall errata on Phoenix
    # (same class as vextract index-0, see 2026-03-24 investigation).
    if name.startswith("VSHIFT"):
        shift_op = op_by_name.get("shift", {})
        if shift_op.get("register_kind") == "scalar":
            shift_reg = regs.get("shift", "r0")
            lines.append(f"  mov r14, #63")
            lines.append(f"  and {shift_reg}, {shift_reg}, r14")
            lines.append(f"  mov r14, #1")
            lines.append(f"  or {shift_reg}, {shift_reg}, r14")

    # VSHUFFLE mod=0 errata: same class of stall as vextract/vshift at 0.
    # OR with 1 ensures the mode register is never zero.
    if name.startswith("VSHUFFLE"):
        mod_op = op_by_name.get("mod", {})
        if mod_op.get("register_kind") == "scalar":
            mod_reg = regs.get("mod", "r0")
            lines.append(f"  mov r14, #1")
            lines.append(f"  or {mod_reg}, {mod_reg}, r14")

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
        align = 32 if kind in ("vector256", "vector512", "accumulator",
                                "wide_y", "sparse_qx", "quad") else 4
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

    # Return sequence: ret lr + NOP sled (5 delay slots on AIE2).
    lines.append("  ret lr")
    lines.extend([f"  nop" for _ in range(5)])
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
    # Resolve misclassified operands so that combo generation assigns correct
    # register names instead of falling through to the "else: 0" branch.
    operands = [_resolve_operand(op) for op in operands]
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
        # Reject SP-relative spill instructions: these need specialized
        # LoadStrategy/StoreStrategy handling but were rejected there due
        # to llvm-mc encoder bugs.  Don't let the fallback pick them up.
        if _is_sp_relative(instr):
            return (False, "SP-relative spill (llvm-mc encoder bug)")
        status, reason = classify_instruction(instr)
        return (status == "testable", reason)

    def generate_test_point(self, instr, regs, in_offset, out_offset, **_kw):
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

        # Reject .tm (tile memory) loads -- these access a special memory
        # region that requires hardware configuration our harness doesn't
        # provide, causing a permanent hardware hang.
        if ".tm" in mnemonic:
            return (False, "tile memory load (.tm) -- needs dedicated harness")

        # Reject conversion loads (ups/conv/unpack) -- llvm-mc has no
        # single-instruction syntax for these.
        if any(s in mnemonic for s in CONVERSION_SUFFIXES):
            return (False, "conversion load (no llvm-mc syntax)")

        operands = instr.get("operands", [])
        sp_relative = _is_sp_relative(instr)

        # Reject ALL SP-relative spill loads: llvm-mc crashes with assertion
        # failure in getSImmOpValueXStep because the encoder step/range
        # requirements don't match the immediate field metadata.  llvm-aie bug.
        # Affects: LDA_dmv_lda_q_ag_spill, VLDA_dmw_lda_{am,w}_ag_spill.
        if sp_relative:
            return (False, "SP-relative spill load (llvm-mc encoder bug)")

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
            # bw=6: am-class quarter registers (amll/amlh/amhl/amhh).
            # These are 256-bit quarters of the 512-bit bml/bmh halves.
            # Loaded/stored via parent half-accumulator using vups/vsrs.
            # Fall through -- handled by standard accumulator infrastructure.

        # Reject unknown operand types (except dontcare and resolvable).
        for op in operands:
            op = _resolve_operand(op)
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

    def generate_test_point(self, instr, regs, in_offset, out_offset, **_kw):
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
    # Check for quad registers FIRST: accumulator with bw=2 is actually
    # quad (q0-q3).  Must check before _effective_kind() which would
    # return "accumulator" and short-circuit.
    kind = op.get("register_kind", "") or ""
    if kind == "accumulator" and op.get("bit_width", 0) == 2:
        return "quad"

    # Then try the composite_register -> TESTABLE_COMPOSITE_KINDS path.
    eff = _effective_kind(op)
    if eff:
        return eff
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

        # Reject .tm (tile memory) stores -- these access a special memory
        # region that requires hardware configuration our harness doesn't
        # provide, causing a permanent hardware hang.
        mnemonic = instr.get("mnemonic", "")
        if ".tm" in mnemonic:
            return (False, "tile memory store (.tm) -- needs dedicated harness")

        # Reject conversion stores (srs/conv/pack) -- llvm-mc has no
        # single-instruction syntax for these.
        if any(s in mnemonic for s in CONVERSION_SUFFIXES):
            return (False, "conversion store (no llvm-mc syntax)")

        operands = instr.get("operands", [])
        sp_relative = _is_sp_relative(instr)

        # Reject ALL SP-relative spill stores: llvm-mc crashes with assertion
        # failure in getSImmOpValueXStep because the encoder step/range
        # requirements don't match the immediate field metadata.  llvm-aie bug.
        # Affects: ST_dmv_sts_q_ag_spill, VST_{128,dmw_sts_{am,w}}_ag_spill.
        if sp_relative:
            return (False, "SP-relative spill store (llvm-mc encoder bug)")

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

        # Reject unknown operand types (except dontcare and resolvable).
        for op in operands:
            op = _resolve_operand(op)
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

    def generate_test_point(self, instr, regs, in_offset, out_offset, **_kw):
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

    DISABLED: AIE2 branch instructions (j, jl, jnz, jz, jnzd, ret) use
    ABSOLUTE byte addresses.  Our assembly is linked at a non-zero offset
    (typically 0xa0+) by aiecc.py, but llvm-mc doesn't support label operands
    for branch targets and emits no relocations.  So `j #N` in our .s file
    becomes an absolute jump to byte N in the final ELF, which lands in the
    objectfifo wrapper code, not our kernel.  Testing branches requires
    either LLVM IR (.ll) tests where the compiler resolves addresses, or
    runtime address computation via register-indirect branches.
    """

    _TESTABLE = frozenset({"j", "jl", "jnz", "jz", "ret", "jnzd"})

    def can_test(self, instr: dict) -> tuple[bool, str]:
        mnemonic = instr.get("mnemonic", "")
        if mnemonic not in self._TESTABLE:
            return (False, "not a branch instruction")
        # All branch instructions disabled in assembly: absolute address
        # targets are unrelocatable.  Tested via .ll batch instead.
        return (False, "branch (tested via .ll, not assembly)")

    def _is_conditional(self, instr: dict) -> bool:
        return instr.get("mnemonic", "") in ("jnz", "jz", "jnzd")

    def _is_indirect(self, instr: dict) -> bool:
        """True if the branch target comes from a pointer register."""
        return any(
            op.get("register_kind") == "pointer"
            for op in instr.get("operands", [])
            if op.get("operand_type") in REGISTER_LIKE_TYPES
        )

    def _is_ret(self, instr: dict) -> bool:
        return instr.get("mnemonic", "") == "ret"

    def compute_input_size(self, instr, regs):
        # Conditional branches use immediate values (mov r0, #N) for the
        # condition, not input buffer loads.  No input buffer needed.
        return 0

    def compute_output_size(self, instr):
        return 8  # 4-byte "before" marker + 4-byte "path" marker

    def generate_combos(self, instr: dict) -> list[dict[str, str]]:
        """Conditional branches get 2 combos, unconditional get 1.

        Combo index 0 = taken path, combo index 1 = not-taken path.
        """
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

        # Two combos: index 0 = taken, index 1 = not-taken.
        return [dict(base_combo), dict(base_combo)]

    def generate_test_point(self, instr, regs, in_offset, out_offset,
                            code_iw_offset=0):
        """Generate branch test with absolute IW addresses.

        Layout (for unconditional j #imm):
            IW+0: mov r14, #170       (before marker)
            IW+1: st r14, [p1, #OUT]  (store before marker)
            IW+2: j #<taken_iw>       (THE BRANCH)
            IW+3..+7: nop (x5)        (delay slots)
            IW+8: mov r14, #187       (fall-through: 0xBB)
            IW+9: st r14, [p1, #OUT+4]
            IW+10: j #<done_iw>       (skip to convergence)
            IW+11..+15: nop (x5)      (delay slots)
            IW+16: mov r14, #204      (taken: 0xCC)
            IW+17: st r14, [p1, #OUT+4]
            done_iw = IW+18

        For register-indirect (j pN, jl pN, ret lr): an extra mov to load
        the target IW address into the pointer/LR register prepends.
        For jnzd: mov counter + mov target pointer prepend.  Combo 0 = taken
        (counter=2, decrements to 1), combo 1 = not-taken (counter=1,
        decrements to 0).
        For conditional (jnz/jz), an extra mov r0, #N prepends.
        Store helpers may emit >1 IW for large offsets (padda sequence).
        We count instructions dynamically to get correct targets.
        """
        name = instr["name"]
        mnemonic = instr.get("mnemonic", "")
        is_cond = self._is_conditional(instr)
        is_ind = self._is_indirect(instr)
        is_ret = self._is_ret(instr)

        BRANCH_DELAY = 5

        # Determine condition value from combo index.
        combo_idx = regs.get("_combo_idx", 0)

        # Phase 1: Build pre-branch instructions and count them.
        pre_branch = []
        pre_branch.append(f"  // ---- test (branch): {name} ----")

        if mnemonic == "jnzd":
            # jnzd $mRx, $mRx0, $mPm: dest = source - 1; jump if nonzero.
            # Combo 0 = taken: source=2 (dec to 1, nonzero).
            # Combo 1 = not-taken: source=1 (dec to 0, zero).
            src_reg = regs.get("mRx0", "r1")
            counter_val = 2 if combo_idx == 0 else 1
            pre_branch.append(f"  mov {src_reg}, #{counter_val}")
        elif is_cond:
            if mnemonic == "jnz":
                cond_val = 1 if combo_idx == 0 else 0
            else:  # jz
                cond_val = 0 if combo_idx == 0 else 1
            pre_branch.append(f"  mov r0, #{cond_val}")

        out_offset = _align(out_offset, 4)
        pre_branch.append(f"  mov r14, #170")
        pre_branch.extend(_scalar_store("r14", "p1", out_offset))

        # For register-indirect and ret, we need a setup mov AFTER building
        # the rest of the layout (to know the target IW), but the mov must
        # appear BEFORE the branch in instruction order.  We insert a
        # placeholder and patch it in phase 4.
        setup_placeholder_idx = None
        if is_ret or is_ind:
            setup_placeholder_idx = len(pre_branch)
            pre_branch.append("  nop")  # placeholder, replaced in phase 4

        # Count actual instruction words (skip comments).
        def _iw_count(lines):
            return sum(1 for l in lines
                       if l.strip() and not l.strip().startswith("//"))

        n_pre = _iw_count(pre_branch)

        # Phase 2: Build fall-through and taken paths, count them.
        fall_through = [f"  mov r14, #187"]
        fall_through.extend(_scalar_store("r14", "p1", out_offset + 4))
        n_fall = _iw_count(fall_through)

        taken = [f"  mov r14, #204"]
        taken.extend(_scalar_store("r14", "p1", out_offset + 4))
        n_taken = _iw_count(taken)

        # Phase 3: Compute absolute IW addresses.
        branch_iw = code_iw_offset + n_pre
        fall_through_start = branch_iw + 1 + BRANCH_DELAY
        taken_start = fall_through_start + n_fall + 1 + BRANCH_DELAY
        done_iw = taken_start + n_taken

        # Patch the setup placeholder with the actual target address.
        if setup_placeholder_idx is not None:
            if is_ret:
                pre_branch[setup_placeholder_idx] = \
                    f"  mov lr, #{taken_start}"
            elif is_ind:
                ptr_reg = regs.get("mPm", "p2")
                pre_branch[setup_placeholder_idx] = \
                    f"  mov {ptr_reg}, #{taken_start}"
                # jnzd also needs pointer setup (already has counter from above).
                if mnemonic == "jnzd":
                    # The placeholder was for the pointer; counter mov is
                    # already in pre_branch from the jnzd block above.
                    pass

        # Phase 4: Assemble the full test point.
        lines = list(pre_branch)

        # The branch instruction.
        if is_ret:
            lines.append(f"  ret lr")
        elif mnemonic == "jnzd":
            dst_reg = regs.get("mRx", "r0")
            src_reg = regs.get("mRx0", "r1")
            ptr_reg = regs.get("mPm", "p2")
            lines.append(f"  jnzd {dst_reg}, {src_reg}, {ptr_reg}")
        elif is_ind:
            ptr_reg = regs.get("mPm", "p2")
            if mnemonic == "j":
                lines.append(f"  j {ptr_reg}")
            elif mnemonic == "jl":
                lines.append(f"  jl {ptr_reg}")
        elif mnemonic == "j":
            lines.append(f"  j #{taken_start}")
        elif mnemonic == "jl":
            lines.append(f"  jl #{taken_start}")
        elif mnemonic == "jnz":
            lines.append(f"  jnz r0, #{taken_start}")
        elif mnemonic == "jz":
            lines.append(f"  jz r0, #{taken_start}")

        lines.extend(_nop_sled(BRANCH_DELAY))

        # Fall-through path (branch NOT taken).
        lines.extend(fall_through)
        lines.append(f"  j #{done_iw}")
        lines.extend(_nop_sled(BRANCH_DELAY))

        # Taken path.
        lines.extend(taken)

        # No explicit convergence marker -- the next test point (or ret)
        # follows at done_iw.
        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LockStrategy: tests lock acquire/release with marker-based verification
# ---------------------------------------------------------------------------

class LockStrategy(TestStrategy):
    """Tests lock instructions using marker-based verification.

    Lock instructions (acq, rel, acq.cond, rel.cond) are side-effect-only:
    they modify lock hardware state but don't produce register outputs.
    We verify execution by storing before/after markers -- if the core
    doesn't stall, both markers appear in the output buffer.

    For acq instructions, a rel is emitted first to set the lock to a
    known value, preventing the core from stalling indefinitely.

    Conditional variants (acq.cond, rel.cond) use r26 as the condition
    register; we set r26=1 so the operation always executes.
    """

    def can_test(self, instr: dict) -> tuple[bool, str]:
        mnemonic = instr.get("mnemonic", "")
        if mnemonic not in LOCK_MNEMONICS:
            return (False, "not a lock instruction")
        return (True, "")

    def _is_acquire(self, instr: dict) -> bool:
        return instr.get("mnemonic", "").startswith("acq")

    def _is_conditional(self, instr: dict) -> bool:
        return ".cond" in instr.get("mnemonic", "")

    def compute_input_size(self, instr, regs):
        return 0  # No input buffer needed.

    def compute_output_size(self, instr):
        return 8  # 4-byte "before" marker + 4-byte "after" marker.

    def generate_combos(self, instr: dict) -> list[dict[str, str]]:
        """Lock instructions get 1 combo with default register values.

        Uses distinct registers for mRx (lock ID) and mRy (value) to
        avoid clobbering during setup.
        """
        combo = {}
        scalar_idx = 0
        # Rotate through scalar registers to avoid using the same register
        # for both the lock ID and the lock value operands.
        scalar_choices = ["r0", "r1", "r2", "r3"]
        for op in instr.get("operands", []):
            op_name = op["name"]
            op_type = op.get("operand_type", "")
            if op_type == "immediate":
                # Use lock ID 0 -- safe default, exists on all tiles.
                combo[op_name] = "0"
            elif op_type in REGISTER_LIKE_TYPES:
                combo[op_name] = scalar_choices[scalar_idx % len(scalar_choices)]
                scalar_idx += 1
            elif op_type in ("unknown", "dontcare"):
                combo[op_name] = "0"
            else:
                combo[op_name] = "0"
        return [combo]

    def generate_test_point(self, instr, regs, in_offset, out_offset, **_kw):
        """Generate lock test with before/after markers.

        Layout:
            mov r14, #170           (before marker)
            st r14, [p1, #OUT]      (store before)
            [mov r26, #1]           (if conditional: enable operation)
            [mov rN, #1]            (set lock value register)
            [rel #id/rX, rN]        (if acquire: pre-set lock so it won't stall)
            <lock instruction>      (THE INSTRUCTION UNDER TEST)
            mov r14, #204           (after marker)
            st r14, [p1, #OUT+4]    (store after -- proves no stall)
        """
        name = instr["name"]
        mnemonic = instr.get("mnemonic", "")
        asm_str = instr.get("asm_string", "")

        lines = []
        lines.append(f"  // ---- test (lock): {name} ----")

        # Store "before" marker.
        out_offset = _align(out_offset, 4)
        lines.append(f"  mov r14, #170")
        lines.extend(_scalar_store("r14", "p1", out_offset))

        # For conditional variants: set r26=1 so the operation executes.
        if self._is_conditional(instr):
            lines.append(f"  mov r26, #1")

        # Determine the lock ID and value register from the combo.
        # Immediate form: id=lock_id, mRy=value register.
        # Register form: mRx=lock_id register, mRy=value register.
        val_reg = regs.get("mRy", "r1")

        # Set the value register to 1.
        lines.append(f"  mov {val_reg}, #1")

        # For register-indirect lock ID, set the lock ID register.
        lock_id_reg = regs.get("mRx", None)
        if lock_id_reg:
            lines.append(f"  mov {lock_id_reg}, #0")

        # For acquire: emit rel first to ensure the lock is acquirable.
        if self._is_acquire(instr):
            if lock_id_reg:
                lines.append(f"  rel {lock_id_reg}, {val_reg}")
            else:
                lock_id_imm = regs.get("id", "0")
                lines.append(f"  rel #{lock_id_imm}, {val_reg}")

        # Execute the instruction under test.
        asm_line = _substitute_asm(asm_str, regs)
        lines.append(f"  {asm_line}")

        # Store "after" marker (proves no stall).
        lines.append(f"  mov r14, #204")
        lines.extend(_scalar_store("r14", "p1", out_offset + 4))

        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# FifoLoadStrategy: tests compressed/sparse FIFO load instructions
# ---------------------------------------------------------------------------

# Mnemonics for FIFO load operations.
_FIFO_PREFIXES = ("vldb.compr", "vldb.sparse")

# QXHLb composite register class: 320-bit sparse registers (quad + vector half).
# mQXHLb = {qwl0, qwl2, qwl1, qwl3, qwh0, qwh2, qwh1, qwh3}
_QXHLB_REGISTER_NAMES = ["qwl0", "qwl1", "qwl2", "qwl3",
                          "qwh0", "qwh1", "qwh2", "qwh3"]


class FifoLoadStrategy(TestStrategy):
    """Tests compressed/sparse FIFO load instructions with markers.

    FIFO operations follow a protocol: RESET -> FILL -> PEEK/POP.
    FILL reads from memory into a hardware decompression FIFO; PEEK/POP
    read decompressed data from the FIFO.

    For FILL/RESET: marker-only verification (no register output).
    For PEEK/POP: RESET + FILL to prime the FIFO, then execute.
    All 16 variants use before/after markers to prove execution
    completed without stalling.

    The pointer register (p2) is aimed at the input buffer so FILL
    reads valid memory.  The FIFO decompresses whatever bytes are
    there; since EMU and HW see identical input data, their outputs
    (including any garbage from "decompressing" random data) match.
    """

    _PEEK_POP_OPS = frozenset({"peek", "pop"})

    def can_test(self, instr: dict) -> tuple[bool, str]:
        mnemonic = instr.get("mnemonic", "")
        if not any(mnemonic.startswith(p) for p in _FIFO_PREFIXES):
            return (False, "not a FIFO load instruction")
        return (True, "")

    def _is_peek_or_pop(self, instr: dict) -> bool:
        """True if the instruction reads from the FIFO (needs setup)."""
        mnemonic = instr.get("mnemonic", "")
        return any(f".{op}" in mnemonic for op in self._PEEK_POP_OPS)

    def _fifo_setup_prefix(self, instr: dict) -> tuple[str, str]:
        """Return (reset_mnemonic, fill_mnemonic) for the instruction's FIFO.

        Compressed: vldb.compr.reset, vldb.compr.fill
        Sparse:     vldb.sparse.reset.N, vldb.sparse.fill.N
        """
        mnemonic = instr.get("mnemonic", "")
        if mnemonic.startswith("vldb.compr"):
            return ("vldb.compr.reset", "vldb.compr.fill")
        # Sparse: mnemonic = vldb.sparse.{op}.{width}
        parts = mnemonic.split(".")
        width = parts[-1] if len(parts) >= 4 else "8"
        return (f"vldb.sparse.reset.{width}", f"vldb.sparse.fill.{width}")

    def compute_input_size(self, instr, regs):
        return 0  # No input buffer needed (markers only).

    def compute_output_size(self, instr):
        return 8  # 4-byte "before" marker + 4-byte "after" marker.

    def generate_combos(self, instr: dict) -> list[dict[str, str]]:
        """One combo with default register values."""
        combo = {}
        for op in instr.get("operands", []):
            op_name = op["name"]
            op_type = op.get("operand_type", "")
            kind = op.get("register_kind", "")
            if op_type in REGISTER_LIKE_TYPES:
                if kind == "pointer":
                    combo[op_name] = "p2"
                elif kind == "vector256":
                    combo[op_name] = "wl0"
                elif kind == "QXHLb":
                    combo[op_name] = _QXHLB_REGISTER_NAMES[0]
                else:
                    names = register_names(kind)
                    combo[op_name] = names[0] if names else "r0"
            elif op_type == "composite_register":
                if kind == "QXHLb":
                    combo[op_name] = _QXHLB_REGISTER_NAMES[0]
                else:
                    combo[op_name] = "wl0"
            else:
                combo[op_name] = "0"
        return [combo]

    def generate_test_point(self, instr, regs, in_offset, out_offset, **_kw):
        """Generate FIFO load test with before/after markers.

        Layout for PEEK/POP:
            padda p2, p0, #in_offset       (point at input data)
            mov r14, #170                  (before marker)
            st r14, [p1, #OUT]
            vldb.{compr|sparse.N}.reset [p2]  (clear FIFO state)
            vldb.{compr|sparse.N}.fill [p2]   (fill from memory)
            nop x5                         (wait for fill pipeline)
            <instruction under test>       (peek/pop)
            mov r14, #204                  (after marker)
            st r14, [p1, #OUT+4]

        Layout for FILL/RESET:
            padda p2, p0, #in_offset       (valid pointer for FILL)
            mov r14, #170                  (before marker)
            st r14, [p1, #OUT]
            <instruction under test>       (fill/reset)
            mov r14, #204                  (after marker)
            st r14, [p1, #OUT+4]
        """
        name = instr["name"]
        mnemonic = instr.get("mnemonic", "")
        asm_str = instr.get("asm_string", "")
        reset_mnem, fill_mnem = self._fifo_setup_prefix(instr)

        FIFO_LATENCY = 5

        lines = []
        lines.append(f"  // ---- test (fifo): {name} ----")

        # Set up pointer to input buffer (valid memory for FILL).
        ptr_reg = regs.get("ptr", "p2")
        lines.extend(_padda_sequence(ptr_reg, "p0", _align(in_offset, 32)))

        # Store "before" marker.
        out_offset = _align(out_offset, 4)
        lines.append(f"  mov r14, #170")
        lines.extend(_scalar_store("r14", "p1", out_offset))

        # For PEEK/POP: prime the FIFO with RESET + FILL + wait.
        if self._is_peek_or_pop(instr):
            lines.append(f"  {reset_mnem} [{ptr_reg}]")
            lines.append(f"  {fill_mnem} [{ptr_reg}]")
            lines.extend(_nop_sled(FIFO_LATENCY))

        # Execute the instruction under test.
        asm_line = _substitute_asm(asm_str, regs)
        lines.append(f"  {asm_line}")

        # Store "after" marker (proves no stall).
        lines.append(f"  mov r14, #204")
        lines.extend(_scalar_store("r14", "p1", out_offset + 4))

        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CascadeStrategy: tests cascade write instruction with markers
# ---------------------------------------------------------------------------

class CascadeStrategy(TestStrategy):
    """Tests cascade write instruction (vmov MCD, $src).

    Cascade writes push data to the MCD FIFO.  However, without a
    downstream consumer tile connected via aie.cascade_flow, even a
    single write stalls the core indefinitely (confirmed on real NPU --
    batch_62 hang investigation, 2026-03-24).  Cascade writes therefore
    cannot be tested in single-tile harness batches.

    Both cascade WRITES and READS are rejected here.  Multi-tile
    cascade_pair infrastructure (CascadeReadStrategy) is needed.
    """

    def can_test(self, instr: dict) -> tuple[bool, str]:
        asm = instr.get("asm_string", "")
        if "MCD" in asm:
            return (False, "cascade write (stalls without downstream consumer)")
        if "SCD" in asm:
            return (False, "cascade read (stalls without neighboring tile)")
        return (False, "not a cascade instruction")

    def compute_input_size(self, instr, regs):
        return 0

    def compute_output_size(self, instr):
        return 8  # before + after markers

    def generate_combos(self, instr: dict) -> list[dict[str, str]]:
        combo = {}
        for op in instr.get("operands", []):
            op_name = op["name"]
            kind = op.get("register_kind", "")
            if kind in ("MvBMXDst", "MvBMXSrc"):
                combo[op_name] = "x0"
            else:
                names = register_names(kind)
                combo[op_name] = names[0] if names else "r0"
        return [combo]

    def generate_test_point(self, instr, regs, in_offset, out_offset, **_kw):
        """Generate cascade write test with markers.

        Layout:
            vlda wl0, [p0, #0]         (load data into x0 low half)
            vlda wh0, [p0, #32]        (load data into x0 high half)
            nop x5                     (wait for load pipeline)
            mov crMCDEn, #1            (enable cascade output)
            mov r14, #170              (before marker)
            st r14, [p1, #OUT]
            vmov MCD, x0              (THE INSTRUCTION)
            mov r14, #204              (after marker)
            st r14, [p1, #OUT+4]
        """
        name = instr["name"]
        asm_str = instr.get("asm_string", "")
        src_reg = regs.get("src", "x0")

        LOAD_LATENCY = 5

        lines = []
        lines.append(f"  // ---- test (cascade): {name} ----")

        # Load known data into the source vector register from input buffer.
        # Use _padda_sequence for large offsets (vlda max immediate is 992).
        in_offset = _align(in_offset, 32)
        if in_offset <= MAX_VECTOR_OFFSET:
            lines.append(f"  vlda wl0, [p0, #{in_offset}]")
            lines.append(f"  vlda wh0, [p0, #{in_offset + 32}]")
        else:
            lines.extend(_padda_sequence("p6", "p0", in_offset))
            lines.append(f"  vlda wl0, [p6, #0]")
            lines.append(f"  vlda wh0, [p6, #32]")
        lines.extend(_nop_sled(LOAD_LATENCY))

        # Enable cascade output.
        lines.append(f"  mov crMCDEn, #1")

        # Store "before" marker.
        out_offset = _align(out_offset, 4)
        lines.append(f"  mov r14, #170")
        lines.extend(_scalar_store("r14", "p1", out_offset))

        # Execute the instruction under test.
        asm_line = _substitute_asm(asm_str, regs)
        lines.append(f"  {asm_line}")

        # Store "after" marker (proves no stall).
        lines.append(f"  mov r14, #204")
        lines.extend(_scalar_store("r14", "p1", out_offset + 4))

        lines.append("")
        return "\n".join(lines)


class CascadeReadStrategy(TestStrategy):
    """Tests cascade read instructions via multi-tile producer-consumer pairs.

    Cascade reads (vmov $dst, SCD / vmov.hi / vmov.lo) stall without data
    from a neighboring tile.  This strategy generates TWO assembly programs:
    a producer (loads data, writes MCD) and a consumer (reads SCD, stores
    result).  The MLIR generator places them in the same column with
    aie.cascade_flow linking row 3 (producer) to row 2 (consumer).

    These are NOT bin-packed into mega-program batches.  Each cascade read
    instruction becomes a standalone cascade_pair batch.
    """

    def can_test(self, instr: dict) -> tuple[bool, str]:
        asm = instr.get("asm_string", "")
        if "SCD" in asm:
            return (True, "")
        return (False, "not a cascade read instruction")

    def _is_half(self, instr: dict) -> bool:
        """True for vmov.hi/vmov.lo (256-bit half reads)."""
        mnemonic = instr.get("mnemonic", "")
        return mnemonic in ("vmov.hi", "vmov.lo")

    def generate_combos(self, instr: dict) -> list[dict[str, str]]:
        # vmov.hi/vmov.lo write to cm (accumulator) registers, not x (vector).
        if self._is_half(instr):
            return [{"dst": "cm0"}]
        return [{"dst": "x0"}]

    def compute_producer_input_size(self) -> int:
        return 64

    def compute_producer_output_size(self) -> int:
        return 8

    def compute_consumer_input_size(self) -> int:
        return 0

    def compute_consumer_output_size(self, instr: dict) -> int:
        # Half variants use marker-based verification (2 x 4-byte markers).
        # Full vmov stores the 512-bit (64-byte) vector result.
        return 8 if self._is_half(instr) else 64

    def compute_input_size(self, instr, regs):
        return self.compute_producer_input_size()

    def compute_output_size(self, instr):
        return self.compute_producer_output_size() + self.compute_consumer_output_size(instr)

    def generate_cascade_pair(self, instr: dict, regs: dict) -> dict:
        """Generate producer and consumer assembly programs.

        Returns dict with 'producer_asm' and 'consumer_asm' strings.
        """
        return {
            "producer_asm": self._generate_producer(),
            "consumer_asm": self._generate_consumer(instr, regs),
        }

    def _generate_producer(self) -> str:
        """Generic producer: load data from p0, write to MCD, markers to p1."""
        LOAD_LATENCY = 5
        lines = []
        lines.append("  // ---- cascade producer: load + write MCD ----")
        lines.append("  vlda wl0, [p0, #0]")
        lines.append("  vlda wh0, [p0, #32]")
        lines.extend(_nop_sled(LOAD_LATENCY))
        lines.append("  mov r14, #170")
        lines.extend(_scalar_store("r14", "p1", 0))
        lines.append("  mov r14, #1")
        lines.append("  mov crMCDEn, r14")
        lines.append("  vmov MCD, x0")
        lines.append("  mov r14, #204")
        lines.extend(_scalar_store("r14", "p1", 4))
        lines.append("")
        return "\n".join(lines)

    def _generate_consumer(self, instr: dict, regs: dict) -> str:
        """Consumer: enable SCD, read cascade, store result to p0.

        For full vmov (512-bit, dst=x0): stores the vector data directly.
        For half variants vmov.hi/vmov.lo (dst=cm0): uses marker-based
        verification (0xAA before, 0xCC after) since accumulator registers
        cannot be stored with plain vst.
        """
        mnemonic = instr.get("mnemonic", "")
        asm_str = instr.get("asm_string", "")
        name = instr["name"]
        lines = []
        lines.append(f"  // ---- cascade consumer: {name} ----")
        if self._is_half(instr):
            # Marker-based: store 0xAA, execute cascade read, store 0xCC.
            lines.append("  mov r14, #170")
            lines.extend(_scalar_store("r14", "p0", 0))
            lines.append("  mov r14, #1")
            lines.append("  mov crSCDEn, r14")
            asm_line = _substitute_asm(asm_str, regs)
            lines.append(f"  {asm_line}")
            lines.append("  mov r14, #204")
            lines.extend(_scalar_store("r14", "p0", 4))
        else:
            lines.append("  mov r14, #1")
            lines.append("  mov crSCDEn, r14")
            asm_line = _substitute_asm(asm_str, regs)
            lines.append(f"  {asm_line}")
            lines.append("  vst wl0, [p0, #0]")
            lines.append("  vst wh0, [p0, #32]")
        lines.append("")
        return "\n".join(lines)

    def generate_test_point(self, instr, regs, in_offset, out_offset, **_kw):
        raise NotImplementedError(
            "CascadeReadStrategy uses generate_cascade_pair(), not generate_test_point()"
        )


# ---------------------------------------------------------------------------
# StreamStrategy: stream read, write, and status instructions
# ---------------------------------------------------------------------------

class StreamStrategy(TestStrategy):
    """Handle stream-related instructions in three internal modes:

    - stream_write: instructions that write to ms (master stream port)
    - stream_read:  mov.d1-d6 with stream source ss0
    - ss_status:    mov $mRa, SS (stream switch status read)
    """

    def _detect_mode(self, instr: dict) -> str:
        """Detect which stream sub-mode applies to *instr*.

        Order matters: SS status check must come BEFORE the ms check
        because mov.nb appears in both SS reads and stream writes.
        """
        asm = instr.get("asm_string", "")
        mnemonic = instr.get("mnemonic", "")
        # SS status -- check BEFORE mnemonic/ms (mov.nb collision)
        if "SS" in asm and "mov" in mnemonic:
            return "ss_status"
        # Stream writes -- word-boundary check for "ms"
        if re.search(r'\bms\b', asm):
            return "stream_write"
        # Stream reads -- mov.d*
        if re.match(r"mov\.d[1-6]", mnemonic):
            return "stream_read"
        return ""

    def can_test(self, instr: dict) -> tuple[bool, str]:
        # doTlast_reg variants have $tlast in asm_string but llvm-mc
        # doesn't support the extra operand.  Skip them -- the base
        # and mnemonic-tlast variants cover the same encodings.
        if self._has_tlast_reg(instr):
            return (False, "doTlast_reg variant (unsupported by llvm-mc)")
        mode = self._detect_mode(instr)
        if mode:
            return (True, "")
        return (False, "not a stream instruction")

    def _has_tlast_reg(self, instr: dict) -> bool:
        return "$tlast" in instr.get("asm_string", "")

    def generate_combos(self, instr: dict) -> list[dict[str, str]]:
        mode = self._detect_mode(instr)
        asm = instr.get("asm_string", "")
        if mode == "ss_status":
            return [{"mRa": "r0"}]
        if mode == "stream_read":
            return [{"dst": "r0", "src": "srSS0"}]
        # stream_write
        combo: dict[str, str] = {}
        if "cph" in asm:
            combo = {"addr": "m0", "nw": "0", "op": "0", "id": "r0"}
        elif "ph" in asm:
            combo = {"id": "r0", "pcktType": "0"}
        else:
            combo = {"src": "r0"}
        if self._has_tlast_reg(instr):
            combo["tlast"] = "r1"
        return [combo]

    def compute_producer_input_size(self, instr: dict) -> int:
        return 0

    def compute_producer_output_size(self, instr: dict) -> int:
        return 8

    def compute_consumer_output_size(self, instr: dict) -> int:
        if self._detect_mode(instr) == "ss_status":
            return 12
        return 4

    def compute_input_size(self, instr: dict, regs: dict[str, str]) -> int:
        return 0

    def compute_output_size(self, instr: dict) -> int:
        return self.compute_producer_output_size(instr) + self.compute_consumer_output_size(instr)

    def generate_test_point(self, instr, regs, in_offset, out_offset, **_kw):
        raise NotImplementedError("StreamStrategy uses generate_stream_pair()")

    def _delay_count(self, instr):
        """Extract delay count from mov.d* mnemonic."""
        m = re.match(r"mov\.d(\d)", instr.get("mnemonic", ""))
        return int(m.group(1)) if m else 1

    def generate_stream_pair(self, instr, regs):
        """Generate producer + consumer assembly pair for a stream instruction.

        Returns dict with 'producer_asm' and 'consumer_asm' keys.
        """
        mode = self._detect_mode(instr)
        if mode == "stream_write":
            return {"producer_asm": self._gen_write_producer(instr, regs),
                    "consumer_asm": self._gen_write_consumer()}
        elif mode == "stream_read":
            return {"producer_asm": self._gen_read_producer(),
                    "consumer_asm": self._gen_read_consumer(instr, regs)}
        else:  # ss_status
            return {"producer_asm": self._gen_status_producer(),
                    "consumer_asm": self._gen_status_consumer(instr, regs)}

    def _gen_write_producer(self, instr, regs):
        """Test tile: immediate value, markers, stream write instruction."""
        asm_str = instr.get("asm_string", "")
        name = instr["name"]
        lines = [f"  // ---- stream write producer (test): {name} ----"]
        lines.append("  mov r0, #0xBEEF")
        if self._has_tlast_reg(instr):
            lines.append("  mov r1, #1")
        if "cph" in asm_str:
            lines.append("  mov m0, #0")
        lines.append("  mov r14, #170")
        lines.extend(_scalar_store("r14", "p0", 0))
        asm_line = _substitute_asm(asm_str, regs)
        lines.append(f"  {asm_line}")
        lines.append("  mov r14, #204")
        lines.extend(_scalar_store("r14", "p0", 4))
        lines.append("")
        return "\n".join(lines)

    def _gen_write_consumer(self):
        """Helper: drain stream via mov.d1, store received value."""
        lines = ["  // ---- stream write consumer (helper): drain stream ----"]
        lines.append("  mov.d1 r0, srSS0")
        lines.append("  nop")
        lines.extend(_scalar_store("r0", "p0", 0))
        lines.append("")
        return "\n".join(lines)

    def _gen_read_producer(self):
        """Helper: write known value to ms."""
        lines = ["  // ---- stream read producer (helper): write ms ----"]
        lines.append("  mov r0, #0xBEEF")
        lines.append("  mov r14, #170")
        lines.extend(_scalar_store("r14", "p0", 0))
        lines.append("  mov ms, r0")
        lines.append("  mov r14, #204")
        lines.extend(_scalar_store("r14", "p0", 4))
        lines.append("")
        return "\n".join(lines)

    def _gen_read_consumer(self, instr, regs):
        """Test tile: execute mov.d* stream read, NOP sled, store result."""
        name = instr["name"]
        delay = self._delay_count(instr)
        asm_str = instr.get("asm_string", "")
        lines = [f"  // ---- stream read consumer (test): {name} ----"]
        asm_line = _substitute_asm(asm_str, regs)
        lines.append(f"  {asm_line}")
        lines.extend(_nop_sled(delay))
        lines.extend(_scalar_store("r0", "p0", 0))
        lines.append("")
        return "\n".join(lines)

    def _gen_status_producer(self):
        """Helper: write dummy value to establish active flow."""
        lines = ["  // ---- SS status producer (helper): establish flow ----"]
        lines.append("  mov r0, #0xDEAD")
        lines.append("  mov r14, #170")
        lines.extend(_scalar_store("r14", "p0", 0))
        lines.append("  mov ms, r0")
        lines.append("  mov r14, #204")
        lines.extend(_scalar_store("r14", "p0", 4))
        lines.append("")
        return "\n".join(lines)

    def _gen_status_consumer(self, instr, regs):
        """Test tile: markers around SS read, plus store status value."""
        name = instr["name"]
        asm_str = instr.get("asm_string", "")
        lines = [f"  // ---- SS status consumer (test): {name} ----"]
        lines.append("  mov r14, #170")
        lines.extend(_scalar_store("r14", "p0", 0))
        asm_line = _substitute_asm(asm_str, regs)
        lines.append(f"  {asm_line}")
        lines.append("  mov r14, #204")
        lines.extend(_scalar_store("r14", "p0", 4))
        lines.extend(_scalar_store("r0", "p0", 8))
        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ConversionStrategy: LLVM IR generation for fused conversion instructions
# ---------------------------------------------------------------------------

def _conversion_base_mnemonic(mnemonic: str) -> str:
    """Strip .2d/.3d addressing mode prefixes to get the base conversion mnemonic.

    Examples:
        "vlda.2d.ups.s32.s16" -> "vlda.ups.s32.s16"
        "vlda.3d.ups.s32.s16" -> "vlda.ups.s32.s16"
        "vlda.ups.s32.s16"    -> "vlda.ups.s32.s16"
        "vst.srs.s16.s32"     -> "vst.srs.s16.s32"
    """
    return mnemonic.replace(".2d.", ".").replace(".3d.", ".")


# UPS/SRS intrinsics that serve as the inverse (round-trip) partner.
# For UPS test points: we call UPS (load+upshift), then need SRS to
# convert back to a vector for storing.
# For SRS test points: we call UPS first to get data into accumulator,
# then call SRS (shift+store).
#
# Keys are the accumulator intrinsic suffix (acc32/acc64), values are
# the corresponding SRS/UPS intrinsic info for round-trip.
_UPS_SRS_PARTNERS = {
    # acc32 UPS -> acc32 SRS (16-bit vector round-trip)
    "acc32.v16.I256.ups": {
        "srs_intrinsic": "I256.v16.acc32.srs",
        "vec_type": "<16 x i16>",
        "acc_type": "<8 x i64>",
    },
    "acc32.v32.I256.ups": {
        "srs_intrinsic": "I256.v32.acc32.srs",
        "vec_type": "<32 x i8>",
        "acc_type": "<16 x i64>",
    },
    # acc64 UPS -> acc64 SRS
    "acc64.v8.I256.ups": {
        "srs_intrinsic": "I256.v8.acc64.srs",
        "vec_type": "<8 x i32>",
        "acc_type": "<8 x i64>",
    },
    "acc64.v16.I256.ups": {
        "srs_intrinsic": "I256.v16.acc64.srs",
        "vec_type": "<16 x i16>",
        "acc_type": "<16 x i64>",
    },
    # acc32 SRS -> acc32 UPS (reverse: need UPS to get into acc first)
    "I256.v16.acc32.srs": {
        "ups_intrinsic": "acc32.v16.I256.ups",
        "vec_type": "<16 x i16>",
        "acc_type": "<8 x i64>",
    },
    "I256.v32.acc32.srs": {
        "ups_intrinsic": "acc32.v32.I256.ups",
        "vec_type": "<32 x i8>",
        "acc_type": "<16 x i64>",
    },
    # acc64 SRS -> acc64 UPS
    "I256.v8.acc64.srs": {
        "ups_intrinsic": "acc64.v8.I256.ups",
        "vec_type": "<8 x i32>",
        "acc_type": "<8 x i64>",
    },
    "I256.v16.acc64.srs": {
        "ups_intrinsic": "acc64.v16.I256.ups",
        "vec_type": "<16 x i16>",
        "acc_type": "<16 x i64>",
    },
}


def generate_conversion_ll(test_points: list[dict]) -> str:
    """Generate LLVM IR (.ll) for a batch of conversion test points.

    Each test point is a dict with keys: mnemonic, intrinsic, in_type,
    out_type, in_bytes, out_bytes, and optionally sign.

    For UPS: load vector -> call UPS -> call SRS (to get back to vector) -> store
    For SRS: load vector -> call UPS (to get into acc) -> call SRS -> store
    For PACK: load vector -> call pack(vec, sign) -> store
    For UNPACK: load vector -> call unpack(vec, sign) -> store
    For CONV: load bf16/fp32 -> convert -> convert back -> store

    Each test point gets its own internal function to avoid instruction
    selection issues when mixing intrinsic types in a single function.
    test_kernel dispatches to each with the appropriate buffer offsets.

    Returns a complete .ll file as a string.
    """
    lines = []
    lines.append('; Auto-generated by isa-test-gen.py')
    lines.append('; Do not edit manually.')
    lines.append('')
    lines.append('target triple = "aie2"')
    lines.append('')

    # Track all intrinsics used so we can declare them at the end.
    intrinsics_used: dict[str, tuple[str, str]] = {}  # name -> (ret_type, arg_types)

    # Each test point gets its own function to avoid instruction selection
    # issues when mixing intrinsic types (pack/unpack fail when combined
    # with ups/srs/conv in a single function).
    helper_funcs = []  # (func_name, in_bytes_consumed, out_bytes_produced)

    for idx, tp in enumerate(test_points):
        mnemonic = tp["mnemonic"]
        intrinsic = tp["intrinsic"]
        in_type = tp["in_type"]
        out_type = tp["out_type"]
        in_bytes = tp["in_bytes"]
        out_bytes = tp["out_bytes"]
        sign = tp.get("sign", 0)

        func_name = f"@_conv_test_{idx}"
        func_lines = []
        func_lines.append(f'define internal void {func_name}(ptr %in, ptr %out) {{')
        func_lines.append(f'entry:')
        func_lines.append(f'  ; ---- test {idx}: {mnemonic} ----')

        var_counter = 0

        # Determine the category from the mnemonic.
        is_ups = ".ups." in mnemonic
        is_srs = ".srs." in mnemonic
        is_pack = ".pack." in mnemonic
        is_unpack = ".unpack." in mnemonic
        in_consumed = 0
        out_produced = 0

        if is_ups:
            partner = _UPS_SRS_PARTNERS[intrinsic]
            vec_type = in_type
            acc_type = out_type
            srs_intrinsic = partner["srs_intrinsic"]
            ups_name = f"@llvm.aie2.{intrinsic}"
            srs_name = f"@llvm.aie2.{srs_intrinsic}"

            func_lines.append(f'  %v = load volatile {vec_type}, ptr %in, align 32')
            func_lines.append(f'  %a = call {acc_type} {ups_name}({vec_type} %v, i32 0, i32 {sign})')
            func_lines.append(f'  %r = call {vec_type} {srs_name}({acc_type} %a, i32 0, i32 1)')
            func_lines.append(f'  store volatile {vec_type} %r, ptr %out, align 32')

            intrinsics_used[ups_name] = (acc_type, f"{vec_type}, i32, i32")
            intrinsics_used[srs_name] = (vec_type, f"{acc_type}, i32, i32")
            in_consumed = in_bytes
            out_produced = in_bytes  # round-trip: output is vec-sized

        elif is_srs:
            partner = _UPS_SRS_PARTNERS[intrinsic]
            acc_type = in_type
            vec_type = out_type
            ups_intrinsic = partner["ups_intrinsic"]
            ups_name = f"@llvm.aie2.{ups_intrinsic}"
            srs_name = f"@llvm.aie2.{intrinsic}"

            func_lines.append(f'  %v = load volatile {vec_type}, ptr %in, align 32')
            func_lines.append(f'  %a = call {acc_type} {ups_name}({vec_type} %v, i32 0, i32 1)')
            func_lines.append(f'  %r = call {vec_type} {srs_name}({acc_type} %a, i32 0, i32 {sign})')
            func_lines.append(f'  store volatile {vec_type} %r, ptr %out, align 32')

            intrinsics_used[ups_name] = (acc_type, f"{vec_type}, i32, i32")
            intrinsics_used[srs_name] = (vec_type, f"{acc_type}, i32, i32")
            in_consumed = out_bytes
            out_produced = out_bytes

        elif is_pack:
            pack_name = f"@llvm.aie2.{intrinsic}"

            func_lines.append(f'  %v = load volatile {in_type}, ptr %in, align 32')
            func_lines.append(f'  %r = call {out_type} {pack_name}({in_type} %v, i32 {sign})')
            func_lines.append(f'  store volatile {out_type} %r, ptr %out, align 32')

            intrinsics_used[pack_name] = (out_type, f"{in_type}, i32")
            in_consumed = in_bytes
            out_produced = out_bytes

        elif is_unpack:
            unpack_name = f"@llvm.aie2.{intrinsic}"

            func_lines.append(f'  %v = load volatile {in_type}, ptr %in, align 32')
            func_lines.append(f'  %r = call {out_type} {unpack_name}({in_type} %v, i32 {sign})')
            func_lines.append(f'  store volatile {out_type} %r, ptr %out, align 32')

            intrinsics_used[unpack_name] = (out_type, f"{in_type}, i32")
            in_consumed = in_bytes
            out_produced = out_bytes

        elif ".conv." in mnemonic:
            conv_name = f"@llvm.aie2.{intrinsic}"

            if "vlda.conv" in mnemonic:
                back_name = "@llvm.aie2.v16accfloat.to.v16bf16"
                func_lines.append(f'  %v = load volatile {in_type}, ptr %in, align 32')
                func_lines.append(f'  %a = call {out_type} {conv_name}({in_type} %v)')
                func_lines.append(f'  %r = call {in_type} {back_name}({out_type} %a)')
                func_lines.append(f'  store volatile {in_type} %r, ptr %out, align 32')
                intrinsics_used[conv_name] = (out_type, in_type)
                intrinsics_used[back_name] = (in_type, out_type)
                in_consumed = in_bytes
                out_produced = in_bytes
            else:
                ups_name = "@llvm.aie2.v16bf16.to.v16accfloat"
                func_lines.append(f'  %v = load volatile {out_type}, ptr %in, align 32')
                func_lines.append(f'  %a = call {in_type} {ups_name}({out_type} %v)')
                func_lines.append(f'  %r = call {out_type} {conv_name}({in_type} %a)')
                func_lines.append(f'  store volatile {out_type} %r, ptr %out, align 32')
                intrinsics_used[ups_name] = (in_type, out_type)
                intrinsics_used[conv_name] = (out_type, in_type)
                in_consumed = out_bytes
                out_produced = out_bytes

        func_lines.append('  ret void')
        func_lines.append('}')
        func_lines.append('')
        helper_funcs.append((func_name, in_consumed, out_produced,
                             '\n'.join(func_lines)))

    # Emit helper functions first.
    for _, _, _, func_text in helper_funcs:
        lines.append(func_text)

    # Emit test_kernel that dispatches to each helper with offset pointers.
    lines.append('define void @test_kernel(ptr %in, ptr %out) {')
    lines.append('entry:')
    in_offset = 0
    out_offset = 0
    for idx, (func_name, in_consumed, out_produced, _) in enumerate(helper_funcs):
        in_ptr = f"%in_{idx}"
        out_ptr = f"%out_{idx}"
        lines.append(f'  {in_ptr} = getelementptr i8, ptr %in, i64 {in_offset}')
        lines.append(f'  {out_ptr} = getelementptr i8, ptr %out, i64 {out_offset}')
        lines.append(f'  call void {func_name}(ptr {in_ptr}, ptr {out_ptr})')
        in_offset += in_consumed
        out_offset += out_produced
    lines.append('  ret void')
    lines.append('}')
    lines.append('')

    # Declare all intrinsics used.
    for name, (ret_type, arg_types) in sorted(intrinsics_used.items()):
        lines.append(f'declare {ret_type} {name}({arg_types})')

    lines.append('')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Branch test generation via LLVM IR
# ---------------------------------------------------------------------------

def generate_branch_ll() -> tuple[str, list[dict]]:
    """Generate LLVM IR (.ll) for branch instruction tests.

    AIE2 branch instructions (j, jl, jnz, jz, jnzd, ret) use absolute byte
    addresses that llvm-mc cannot relocate.  By generating LLVM IR, the
    compiler handles address resolution at link time.

    Tests exercise:
      - Unconditional branch (j): br label
      - Conditional branch taken (jnz/jz): br i1 %cond, label, label
      - Conditional branch not-taken: same with opposite condition
      - Function call and return (jl/ret): call + ret void
      - Loop with decrement (jnzd): counted loop

    Each test writes a 4-byte marker to the output buffer.  The marker value
    indicates which path executed (0xAA=before, 0xBB=not-taken, 0xCC=taken,
    counter value for loops).

    Returns:
        (ll_content, test_metadata) where test_metadata is a list of dicts
        with instruction, in_offset, in_size, out_offset, out_size.
    """
    lines = []
    lines.append('; Auto-generated by isa-test-gen.py (branch tests)')
    lines.append('; Do not edit manually.')
    lines.append('')
    lines.append('target triple = "aie2"')
    lines.append('')

    helpers = []  # (func_name, in_bytes, out_bytes, func_text, instr_name)

    # --- Test 0: Unconditional branch (j) ---
    helpers.append(("@_branch_j", 0, 8, """\
define internal void @_branch_j(ptr %in, ptr %out) {
entry:
  store volatile i32 170, ptr %out, align 4
  br label %taken
taken:
  %p1 = getelementptr i8, ptr %out, i64 4
  store volatile i32 204, ptr %p1, align 4
  ret void
}
""", "J_jump_imm"))

    # --- Test 1: Conditional branch taken (jnz path) ---
    helpers.append(("@_branch_jnz_taken", 4, 8, """\
define internal void @_branch_jnz_taken(ptr %in, ptr %out) {
entry:
  %cv = load volatile i32, ptr %in, align 4
  %cond = icmp ne i32 %cv, 0
  store volatile i32 170, ptr %out, align 4
  br i1 %cond, label %taken, label %not_taken
taken:
  %p1 = getelementptr i8, ptr %out, i64 4
  store volatile i32 204, ptr %p1, align 4
  br label %done
not_taken:
  %p2 = getelementptr i8, ptr %out, i64 4
  store volatile i32 187, ptr %p2, align 4
  br label %done
done:
  ret void
}
""", "JNZ"))

    # --- Test 2: Conditional branch forced not-taken (hardcoded false) ---
    helpers.append(("@_branch_forced_not_taken", 0, 8, """\
define internal void @_branch_forced_not_taken(ptr %in, ptr %out) {
entry:
  store volatile i32 170, ptr %out, align 4
  br i1 false, label %taken, label %not_taken
taken:
  %p1 = getelementptr i8, ptr %out, i64 4
  store volatile i32 204, ptr %p1, align 4
  br label %done
not_taken:
  %p2 = getelementptr i8, ptr %out, i64 4
  store volatile i32 187, ptr %p2, align 4
  br label %done
done:
  ret void
}
""", "JZ"))

    # --- Test 3: Conditional branch taken (jz path, cond=0) ---
    helpers.append(("@_branch_jz_taken", 4, 8, """\
define internal void @_branch_jz_taken(ptr %in, ptr %out) {
entry:
  %cv = load volatile i32, ptr %in, align 4
  %cond = icmp eq i32 %cv, 0
  store volatile i32 170, ptr %out, align 4
  br i1 %cond, label %taken, label %not_taken
taken:
  %p1 = getelementptr i8, ptr %out, i64 4
  store volatile i32 204, ptr %p1, align 4
  br label %done
not_taken:
  %p2 = getelementptr i8, ptr %out, i64 4
  store volatile i32 187, ptr %p2, align 4
  br label %done
done:
  ret void
}
""", "JZ"))

    # --- Test 4: Function call and return (jl/ret) ---
    helpers.append(("@_branch_jl_ret", 4, 8, """\
define internal i32 @_callee(i32 %x) {
  %r = add i32 %x, 1
  ret i32 %r
}

define internal void @_branch_jl_ret(ptr %in, ptr %out) {
entry:
  %v = load volatile i32, ptr %in, align 4
  store volatile i32 170, ptr %out, align 4
  %r = call i32 @_callee(i32 %v)
  %p1 = getelementptr i8, ptr %out, i64 4
  store volatile i32 %r, ptr %p1, align 4
  ret void
}
""", "JL"))

    # --- Test 5: Counted loop (jnzd) ---
    helpers.append(("@_branch_loop", 4, 8, """\
define internal void @_branch_loop(ptr %in, ptr %out) {
entry:
  %n = load volatile i32, ptr %in, align 4
  store volatile i32 170, ptr %out, align 4
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%i_next, %loop]
  %i_next = add i32 %i, 1
  %done = icmp eq i32 %i_next, %n
  br i1 %done, label %exit, label %loop
exit:
  %p1 = getelementptr i8, ptr %out, i64 4
  store volatile i32 %i_next, ptr %p1, align 4
  ret void
}
""", "JNZD"))

    # --- Test 6: Indirect jump (j pN -- register indirect) ---
    helpers.append(("@_branch_j_ind", 4, 8, """\
define internal void @_branch_j_ind(ptr %in, ptr %out) {
entry:
  %v = load volatile i32, ptr %in, align 4
  store volatile i32 170, ptr %out, align 4
  %cond = icmp sgt i32 %v, 0
  br i1 %cond, label %positive, label %negative
positive:
  br label %merge
negative:
  br label %merge
merge:
  %r = phi i32 [1, %positive], [0, %negative]
  %p1 = getelementptr i8, ptr %out, i64 4
  store volatile i32 %r, ptr %p1, align 4
  ret void
}
""", "J_jump_ind"))

    # --- Test 7: Indirect jl (function pointer call) ---
    helpers.append(("@_branch_jl_ind", 4, 8, """\
define internal i32 @_ind_callee_a(i32 %x) {
  %r = add i32 %x, 100
  ret i32 %r
}

define internal void @_branch_jl_ind(ptr %in, ptr %out) {
entry:
  %v = load volatile i32, ptr %in, align 4
  store volatile i32 170, ptr %out, align 4
  %r = call i32 @_ind_callee_a(i32 %v)
  %p1 = getelementptr i8, ptr %out, i64 4
  store volatile i32 %r, ptr %p1, align 4
  ret void
}
""", "JL_IND"))

    # --- Test 8: Return from nested call (ret) ---
    helpers.append(("@_branch_ret_nested", 4, 8, """\
define internal i32 @_nested_inner(i32 %x) {
  %r = mul i32 %x, 3
  ret i32 %r
}

define internal i32 @_nested_outer(i32 %x) {
  %r1 = call i32 @_nested_inner(i32 %x)
  %r2 = add i32 %r1, 7
  ret i32 %r2
}

define internal void @_branch_ret_nested(ptr %in, ptr %out) {
entry:
  %v = load volatile i32, ptr %in, align 4
  store volatile i32 170, ptr %out, align 4
  %r = call i32 @_nested_outer(i32 %v)
  %p1 = getelementptr i8, ptr %out, i64 4
  store volatile i32 %r, ptr %p1, align 4
  ret void
}
""", "RET"))

    # --- Test 9: Nested branches (diamond pattern) ---
    helpers.append(("@_branch_diamond", 8, 8, """\
define internal void @_branch_diamond(ptr %in, ptr %out) {
entry:
  %a = load volatile i32, ptr %in, align 4
  %p_b = getelementptr i8, ptr %in, i64 4
  %b = load volatile i32, ptr %p_b, align 4
  %c1 = icmp sgt i32 %a, %b
  br i1 %c1, label %left, label %right
left:
  br label %merge
right:
  br label %merge
merge:
  %result = phi i32 [%a, %left], [%b, %right]
  store volatile i32 170, ptr %out, align 4
  %p1 = getelementptr i8, ptr %out, i64 4
  store volatile i32 %result, ptr %p1, align 4
  ret void
}
""", "J_jump_imm"))

    # Emit all helper functions.
    for _, _, _, func_text, _ in helpers:
        lines.append(func_text)

    # Emit test_kernel dispatcher.
    lines.append('define void @test_kernel(ptr %in, ptr %out) {')
    lines.append('entry:')
    in_offset = 0
    out_offset = 0
    meta = []
    for idx, (func_name, in_bytes, out_bytes, _, instr_name) in enumerate(helpers):
        lines.append(f'  %in_{idx} = getelementptr i8, ptr %in, i64 {in_offset}')
        lines.append(f'  %out_{idx} = getelementptr i8, ptr %out, i64 {out_offset}')
        lines.append(f'  call void {func_name}(ptr %in_{idx}, ptr %out_{idx})')
        meta.append({
            "instruction": instr_name,
            "slot": "branch",
            "combo_index": 0,
            "operands": {},
            "in_offset": in_offset,
            "in_size": max(in_bytes, 4),  # minimum 4 bytes for PRNG alignment
            "out_offset": out_offset,
            "out_size": out_bytes,
        })
        in_offset += max(in_bytes, 4)
        out_offset += out_bytes
    lines.append('  ret void')
    lines.append('}')
    lines.append('')

    total_in = in_offset
    total_out = out_offset
    return '\n'.join(lines), meta, total_in, total_out


class ConversionStrategy(TestStrategy):
    """Tests conversion instructions via LLVM IR instead of assembly.

    Conversion instructions (vlda.ups, vst.srs, vst.pack, vldb.unpack)
    cannot be assembled by llvm-mc directly.  Instead, we generate LLVM IR
    that calls Peano intrinsics; the compiler fuses load+intrinsic into
    the fused instruction automatically.

    This strategy does not generate assembly test points.  Instead, it
    identifies conversion instructions and collects them for batch
    processing via generate_conversion_ll().
    """

    def can_handle(self, instr: dict) -> bool:
        """Return True if this instruction is a conversion we handle via LLVM IR."""
        mnemonic = instr.get("mnemonic", "")
        if not any(s in mnemonic for s in _CONVERSION_IR_SUFFIXES):
            return False
        base = _conversion_base_mnemonic(mnemonic)
        if base not in CONVERSION_INTRINSICS:
            return False
        return True

    def can_test(self, instr: dict) -> tuple[bool, str]:
        """ConversionStrategy does not participate in normal strategy dispatch.

        Conversion instructions are handled separately in generate_all().
        This method is not called by classify_with_strategies().
        """
        return (False, "conversion handled via LLVM IR")

    def generate_combos(self, instr: dict) -> list[dict[str, str]]:
        """Return a single empty combo -- intrinsics determine everything."""
        return [{}]

    def compute_input_size(self, instr: dict, regs: dict[str, str]) -> int:
        """Return input size from the conversion intrinsic table."""
        mnemonic = instr.get("mnemonic", "")
        base = _conversion_base_mnemonic(mnemonic)
        info = CONVERSION_INTRINSICS.get(base)
        if info is None:
            return 32
        # For UPS, input is the vector being loaded.
        # For SRS, we load a vector (out_bytes) and UPS it, then SRS.
        if ".srs." in mnemonic:
            return info["out_bytes"]
        return info["in_bytes"]

    def compute_output_size(self, instr: dict) -> int:
        """Return output size from the conversion intrinsic table."""
        mnemonic = instr.get("mnemonic", "")
        base = _conversion_base_mnemonic(mnemonic)
        info = CONVERSION_INTRINSICS.get(base)
        if info is None:
            return 32
        # For UPS, output is vector-sized (round-tripped through SRS).
        if ".ups." in mnemonic:
            return info["in_bytes"]
        return info["out_bytes"]

    def generate_test_point(self, instr, regs, in_offset, out_offset, **_kw):
        """Not used -- conversion batches go through generate_conversion_ll()."""
        raise NotImplementedError(
            "ConversionStrategy does not generate assembly test points; "
            "use generate_conversion_ll() instead"
        )


# ---------------------------------------------------------------------------
# DoneStrategy: tests the `done` instruction (core halt)
# ---------------------------------------------------------------------------


class DoneStrategy(TestStrategy):
    """Tests the `done` instruction using a canary pattern.

    `done` halts the core.  We store a before-marker, execute `done`, then
    store an after-marker.  The after-marker should never appear in the
    output because the core stops at `done`.  Both EMU and HW should produce
    identical output: before=0xAA, after=0x00 (never written).
    """

    def can_test(self, instr: dict) -> tuple[bool, str]:
        if instr.get("mnemonic", "") == "done":
            return (True, "")
        return (False, "not the done instruction")

    def compute_input_size(self, instr, regs):
        return 0

    def compute_output_size(self, instr):
        return 8  # before marker + after marker (canary)

    def generate_combos(self, instr: dict) -> list[dict[str, str]]:
        return [{}]  # No operands.

    def generate_test_point(self, instr, regs, in_offset, out_offset, **_kw):
        lines = []
        lines.append(f"  // ---- test (done): DONE ----")

        # Store before marker.
        out_offset = _align(out_offset, 4)
        lines.append(f"  mov r14, #170")
        lines.extend(_scalar_store("r14", "p1", out_offset))

        # Execute done -- core halts here.
        lines.append(f"  done")

        # After marker (canary -- should never execute).
        lines.append(f"  mov r14, #204")
        lines.extend(_scalar_store("r14", "p1", out_offset + 4))

        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# EventStrategy: tests the `event` instruction
# ---------------------------------------------------------------------------


class EventStrategy(TestStrategy):
    """Tests the `event` instruction using marker-based verification.

    `event $val` fires a hardware event (trace system).  It does not stall
    the core or produce register output.  We verify execution by storing
    before/after markers -- both should appear since event is non-blocking.
    """

    def can_test(self, instr: dict) -> tuple[bool, str]:
        if instr.get("mnemonic", "") == "event":
            return (True, "")
        return (False, "not the event instruction")

    def compute_input_size(self, instr, regs):
        return 0

    def compute_output_size(self, instr):
        return 8  # before + after markers

    def generate_combos(self, instr: dict) -> list[dict[str, str]]:
        return [{"val": "0"}]  # Event value 0 (2-bit field: 0-3).

    def generate_test_point(self, instr, regs, in_offset, out_offset, **_kw):
        lines = []
        lines.append(f"  // ---- test (event): EVENT ----")

        # Store before marker.
        out_offset = _align(out_offset, 4)
        lines.append(f"  mov r14, #170")
        lines.extend(_scalar_store("r14", "p1", out_offset))

        # Execute event (non-blocking, fires trace event).
        val = regs.get("val", "0")
        lines.append(f"  event #{val}")

        # Store after marker (proves event didn't stall).
        lines.append(f"  mov r14, #204")
        lines.extend(_scalar_store("r14", "p1", out_offset + 4))

        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PointerArithStrategy: tests in-place pointer updates (padda/paddb/padds)
# ---------------------------------------------------------------------------

# All non-SP padda/paddb/padds variants.  These are in-place pointer updates
# where the pointer register is both input and output.  ComputeStrategy
# wrongly skips loading the pointer because detect_output_operands marks it
# as output; this strategy loads it explicitly.
_POINTER_ARITH_NAMES = {
    "PADDA_2D", "PADDA_3D",
    "PADDA_lda_ptr_inc_idx", "PADDA_lda_ptr_inc_idx_imm",
    "PADDB_2D", "PADDB_3D",
    "PADDB_ldb_ptr_inc_nospill_nrm", "PADDB_ldb_ptr_inc_nrm_imm",
    "PADDS_2D", "PADDS_3D",
    "PADDS_st_ptr_inc_idx", "PADDS_st_ptr_inc_idx_imm",
}


class PointerArithStrategy(TestStrategy):
    """Tests in-place pointer arithmetic (padda/paddb/padds 2D/3D/idx).

    These instructions update a pointer register in-place:
        padda.2d [$ptr], $mod   -- ptr += step(mod) with 2D wrap
        padda [$ptr], $mod      -- ptr += step(mod)
        padda [$ptr], #imm      -- ptr += imm

    The pointer is BOTH input and output.  We:
      1. Load the pointer value from input buffer into a scalar register
      2. Move it into the pointer register (p2-p5)
      3. Load the modifier register from input (if present)
      4. Execute the instruction
      5. Move the pointer back to a scalar register
      6. Store the scalar to the output buffer
    """

    def can_test(self, instr: dict) -> tuple[bool, str]:
        name = instr.get("name", "")
        if name in _POINTER_ARITH_NAMES:
            return (True, "")
        return (False, "not an in-place pointer arithmetic instruction")

    def compute_input_size(self, instr, regs):
        """Pointer (4 bytes) + modifier (4 bytes) if idx variant.

        2D/3D variants use dimension registers (d0-d7) which are configured
        externally -- no input data needed for those.  idx variants use
        modifier registers (m0-m7) which CAN be loaded from input.
        """
        name = instr.get("name", "")
        is_2d_3d = "2D" in name or "3D" in name
        if is_2d_3d:
            return 4  # Just the pointer value.
        has_mod = any(
            op.get("register_kind") == "modifier_m"
            for op in instr.get("operands", [])
        )
        return 8 if has_mod else 4

    def compute_output_size(self, instr):
        return 4  # One 32-bit pointer value.

    def generate_combos(self, instr: dict) -> list[dict[str, str]]:
        """Generate combos for pointer x modifier (or pointer x immediate).

        2D/3D variants use dimension registers (d0-d7) while idx variants
        use modifier registers (m0-m7).  Both are 3-bit fields encoding
        the same physical register space, but the assembler requires
        different names.
        """
        name = instr.get("name", "")
        operands = instr.get("operands", [])
        combos = []

        ptr_choices = ["p2", "p3", "p4", "p5"]
        is_2d_3d = "2D" in name or "3D" in name
        # 2D/3D: dimension registers d0-d7.  idx: modifier registers m0-m7.
        mod_choices = ["d0", "d1"] if is_2d_3d else ["m0", "m1"]
        # Small immediates that stay in valid memory range.
        imm_choices = ["0", "4", "-4", "32", "64"]

        has_mod = any(op.get("register_kind") == "modifier_m" for op in operands)
        has_imm = any(op.get("name") == "imm" for op in operands)

        if has_mod:
            for ptr in ptr_choices:
                for mod in mod_choices:
                    combo = {"ptr": ptr, "mod": mod}
                    # Fill dontcare fields with "0".
                    for op in operands:
                        if op["name"].startswith("dontcare"):
                            combo[op["name"]] = "0"
                    combos.append(combo)
        elif has_imm:
            for ptr in ptr_choices:
                for imm in imm_choices:
                    combo = {"ptr": ptr, "imm": imm}
                    for op in operands:
                        if op["name"].startswith("dontcare"):
                            combo[op["name"]] = "0"
                    combos.append(combo)
        else:
            # Fallback: just pointer combos.
            for ptr in ptr_choices:
                combo = {"ptr": ptr}
                for op in operands:
                    if op["name"].startswith("dontcare"):
                        combo[op["name"]] = "0"
                combos.append(combo)

        return combos

    def generate_test_point(self, instr, regs, in_offset, out_offset, **_kw):
        lines = []
        name = instr["name"]
        mnemonic = instr["mnemonic"]
        asm_string = instr.get("asm_string", "")
        operands = instr.get("operands", [])
        ptr_reg = regs.get("ptr", "p2")
        mod_reg = regs.get("mod", "m0")

        in_offset = _align(in_offset, 4)
        out_offset = _align(out_offset, 4)

        lines.append(f"  // ---- test (ptr_arith): {name} ptr={ptr_reg} ----")

        # Step 1: Load the pointer value from input buffer into r14,
        # then move to the pointer register.
        lines.extend(_scalar_load("r14", "p0", in_offset))
        lines.extend(_nop_sled(LOAD_LATENCY))
        lines.append(f"  mov {ptr_reg}, r14")

        # Step 2: Load modifier from input (if idx variant uses one).
        # 2D/3D variants use dimension registers (d0-d7) configured externally
        # -- we just select which one via the combo.  idx variants use modifier
        # registers (m0-m7) which can be loaded from input data.
        is_2d_3d = "2D" in name or "3D" in name
        has_mod = (
            not is_2d_3d
            and any(
                op.get("register_kind") == "modifier_m" for op in operands
            )
        )
        if has_mod:
            lines.extend(_scalar_load("r13", "p0", in_offset + 4))
            lines.extend(_nop_sled(LOAD_LATENCY))
            lines.append(f"  mov {mod_reg}, r13")

        # Step 3: Execute the instruction.
        # Reconstruct assembly from asm_string template by substituting
        # operand placeholders ($ptr, $mod, $imm, $dontcareN).
        asm_line = asm_string
        for op_name, op_val in regs.items():
            placeholder = f"${op_name}"
            if op_name == "imm":
                asm_line = asm_line.replace(placeholder, f"#{op_val}")
            elif op_name.startswith("dontcare") or op_name.startswith("_"):
                continue  # Handled below / internal key.
            else:
                asm_line = asm_line.replace(placeholder, str(op_val))
        # Replace any remaining dontcare placeholders not in regs.
        for op in operands:
            if op["name"].startswith("dontcare"):
                asm_line = asm_line.replace(f"${op['name']}", "0")
        lines.append(f"  {asm_line}")

        # Step 4: Move modified pointer to scalar and store to output.
        lines.append(f"  mov r14, {ptr_reg}")
        lines.extend(_scalar_store("r14", "p1", out_offset))

        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PaddaSpStrategy: tests `padda [sp], $imm`
# ---------------------------------------------------------------------------


class PaddaSpStrategy(TestStrategy):
    """Tests `padda [sp], $imm` -- SP pointer increment.

    `padda [sp], $imm` adds imm*32 to SP (p6).  Since SP has no explicit
    output register, we:
      1. Save p6 to a scratch register
      2. Set p6 to a known value (0)
      3. Execute padda [sp], #imm
      4. Read back p6 (now = 0 + imm*32) and store to output
      5. Restore p6

    Only handles padda [sp]; paddb [sp] is blocked by llvm-mc #858.
    """

    def can_test(self, instr: dict) -> tuple[bool, str]:
        name = instr.get("name", "")
        if name == "PADDA_sp_imm":
            return (True, "")
        if name == "PADDB_sp_imm":
            return (False, "paddb [sp] blocked by llvm-mc #858")
        return (False, "not a padda [sp] instruction")

    def compute_input_size(self, instr, regs):
        return 0

    def compute_output_size(self, instr):
        return 4  # One 32-bit pointer value.

    def generate_combos(self, instr: dict) -> list[dict[str, str]]:
        # Use imm=32 (one step).  llvm-mc requires SP immediates to be
        # multiples of 32; the ISA JSON scale=32 means the HW field stores
        # imm/32, but the assembler takes the raw byte value.
        return [{"imm": "32", "dontcare2": "0"}]

    def generate_test_point(self, instr, regs, in_offset, out_offset, **_kw):
        lines = []
        lines.append(f"  // ---- test (padda_sp): PADDA_sp_imm ----")

        out_offset = _align(out_offset, 4)

        # Save current p6 (SP) to r13 so we can restore it later.
        # mov r13, p6 uses the mv slot (MOV_mv_scl).
        lines.append(f"  mov r13, p6")

        # Set p6 to 0 so the result is deterministic.
        lines.append(f"  movxm p6, #0")

        # Execute padda [sp], #imm.
        imm = regs.get("imm", "32")
        lines.append(f"  padda [sp], #{imm}")

        # Read back p6 (now = 0 + imm*32) and store to output.
        lines.append(f"  mov r14, p6")
        lines.extend(_scalar_store("r14", "p1", out_offset))

        # Restore p6.
        lines.append(f"  mov p6, r13")

        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# VmacStrategy: tests MAC/MUL instructions with valid config words
# ---------------------------------------------------------------------------

# All MAC-family mnemonics that use a config register ($c operand).
# These all use the vmac_*_core encoding in AIE2.
_VMAC_MNEMONICS = frozenset({
    "vmac", "vmac.f", "vmul", "vmul.f",
    "vaddmac", "vaddmac.f", "vsubmac",
    "vaddmsc", "vaddmsc.f", "vsubmsc",
    "vmsc", "vmsc.f",
    "vnegmac", "vnegmsc", "vnegmul", "vnegmul.f",
})


def _make_vmac_config(amode: int, bmode: int, variant: int,
                      sgn_x: int, sgn_y: int, zero_acc: int,
                      sub0: int = 0) -> int:
    """Build a MAC config word from individual fields.

    Bit layout (from aietools python_model/model/mulmac.py):
        bit 0:     zero_acc  (1 = clear accumulator before multiply)
        bits 1-2:  amode     (0=acc32, 1=acc64, 2=bf16)
        bits 3-4:  bmode     (element type pair: 0=8x8, 1=16x16, 2=32x32, 3=bf16)
        bits 5-7:  variant   (0=dense, 1=sparse narrow, 2=sparse wide)
        bit 8:     sgn_y     (1=signed Y input)
        bit 9:     sgn_x     (1=signed X input)
        bit 11:    sub0      (1=subtract mode)
    """
    return ((zero_acc << 0) | (amode << 1) | (bmode << 3) | (variant << 5)
            | (sgn_y << 8) | (sgn_x << 9) | (sub0 << 11))


def _vmac_configs_for_instr(name: str) -> list[int]:
    """Return valid config words for a MAC instruction encoding name.

    The encoding name (e.g., "VMAC_vmac_cm_core_dense") tells us:
      - cm_core vs bm_core: accumulator mode (amode)
      - _F_ prefix: floating-point (bf16) mode
      - dense/sparse_narrow/sparse_wide: variant

    We generate a small set of configs covering the main element type
    geometries.  All use zero_acc=1 (clear accumulator, so the output is
    purely from the multiply -- easier to verify).

    Args:
        name: ISA instruction name (e.g., "VMAC_vmac_cm_core_dense").

    Returns:
        List of valid config word integers.
    """
    name_upper = name.upper()

    # Determine variant from encoding name.
    if "SPARSE_WIDE" in name_upper:
        variant = 2
    elif "SPARSE_NARROW" in name_upper:
        variant = 1
    else:
        variant = 0  # dense

    # Determine if this is a floating-point (bf16) instruction.
    # The _F_ prefix in the ISA name (e.g., VMAC_F_vmac_bm_core_dense)
    # or .f mnemonic suffix indicates bf16 mode.
    is_float = "_F_" in name_upper

    # Determine accumulator width from encoding name.
    is_cm = "CM_CORE" in name_upper  # 1024-bit full accumulator (acc64)
    is_bm = "BM_CORE" in name_upper  # 512-bit half accumulator (acc32)

    configs = []

    # Valid (amode, bmode, variant) triples from CONFIG_GEOMETRY_TABLE.
    # Only generate configs that the hardware geometry table supports.
    VALID_DENSE = [
        # amode=0 (acc32)
        (0, 0, 0),  # i8xi4  4x16x8
        (0, 1, 0),  # i8xi8  4x8x8
        (0, 2, 0),  # i16xi8  4x4x8
        (0, 3, 0),  # i16xi16  4x2x8
        # amode=1 (acc64)
        (1, 0, 0),  # i32xi16  4x2x4
        (1, 2, 0),  # i16xi8  2x8x8
        (1, 2, 1),  # i16xi8  4x8x4
        (1, 3, 0),  # i16xi16  2x4x8
        (1, 3, 1),  # i16xi16  4x4x4
    ]
    VALID_SPARSE = [
        # amode=0 (acc32)
        (0, 0, 1),  # i8xi4 sparse
        (0, 1, 5),  # i8xi8 sparse
        # amode=1 (acc64)
        (1, 2, 2),  # i16xi8 sparse
        (1, 3, 5),  # i16xi16 sparse
    ]
    VALID_BF16_DENSE = [(2, 3, 0)]
    VALID_BF16_SPARSE = [(2, 3, 2)]  # bf16 sparse
    VALID_BF16_ELEMWISE = [(2, 3, 1)]  # bf16 element-wise

    if is_float:
        if "SPARSE" in name_upper:
            table = VALID_BF16_SPARSE
        else:
            # Include dense + element-wise for non-sparse bf16.
            table = VALID_BF16_DENSE + VALID_BF16_ELEMWISE
        for amode, bmode, v in table:
            configs.append(_make_vmac_config(
                amode=amode, bmode=bmode, variant=v,
                sgn_x=0, sgn_y=0, zero_acc=1))
    elif is_cm:
        # acc64: amode=1 entries.
        table = VALID_SPARSE if "SPARSE" in name_upper else VALID_DENSE
        for amode, bmode, v in table:
            if amode != 1:
                continue
            for sgn_x, sgn_y in ((1, 1), (0, 0)):
                configs.append(_make_vmac_config(
                    amode=amode, bmode=bmode, variant=v,
                    sgn_x=sgn_x, sgn_y=sgn_y, zero_acc=1))
    elif is_bm:
        # acc32: amode=0 entries.
        table = VALID_SPARSE if "SPARSE" in name_upper else VALID_DENSE
        for amode, bmode, v in table:
            if amode != 0:
                continue
            for sgn_x, sgn_y in ((1, 1), (0, 0)):
                configs.append(_make_vmac_config(
                    amode=amode, bmode=bmode, variant=v,
                    sgn_x=sgn_x, sgn_y=sgn_y, zero_acc=1))
    else:
        # Fallback: one known-good config.
        configs.append(_make_vmac_config(
            amode=1, bmode=3, variant=0,
            sgn_x=1, sgn_y=1, zero_acc=1))

    return configs


_ACCUM_ARITH_MNEMONICS = frozenset({
    "vadd", "vadd.f",
    "vsub", "vsub.f",
    "vneg", "vneg.f",
    "vnegadd", "vnegadd.f",
    "vnegsub", "vnegsub.f",
})


def _accum_arith_configs() -> list[int]:
    """Config words for accumulator add/sub/neg instructions.

    Bit layout (from execute_acc_add_sub):
        bit 0:     zero_acc1  (1 = clear acc1 before operation)
        bit 10:    shift16    (1 = right-shift result by 16)
        bit 11:    sub_acc1   (1 = negate acc1)
        bit 12:    sub_acc2   (1 = negate acc2)

    Config=0 tests the plain operation (no zeroing, no shift, no
    negation).  Additional config values (zero_acc1, shift16, etc.) can
    be added once the basic path is validated.
    """
    return [
        0x0000,  # Plain operation (no zeroing, no shift, no negation)
    ]


class AccumArithStrategy(TestStrategy):
    """Tests accumulator add/sub/neg instructions with valid config words.

    These instructions (vadd, vadd.f, vsub, vneg, etc.) take a scalar
    config register ($c) that controls zero_acc1, shift16, sub_acc1,
    sub_acc2 flags.  Without a valid config word, random data in the
    config register causes undefined behavior (e.g., zeroing an
    accumulator, negating an operand, or shifting the result).

    Uses the same test-point generation pattern as VmacStrategy:
    load inputs -> set config via mov -> execute -> store outputs.
    """

    def can_test(self, instr: dict) -> tuple[bool, str]:
        mnemonic = instr.get("mnemonic", "")
        if mnemonic not in _ACCUM_ARITH_MNEMONICS:
            return (False, "not an accumulator arithmetic instruction")
        if _is_sp_relative(instr):
            return (False, "SP-relative (unexpected)")
        status, reason = classify_instruction(instr)
        if status != "testable":
            return (False, reason)
        return (True, "")

    def _find_config_operand(self, instr: dict) -> Optional[str]:
        """Find the config register operand name (usually 'c')."""
        for op in instr.get("operands", []):
            if op["name"] == "c" and op.get("register_kind") == "scalar":
                return "c"
        return None

    def generate_combos(self, instr: dict) -> list[dict[str, str]]:
        """Generate combos pairing register assignments with valid configs."""
        base_combos = generate_operand_combos(instr)
        if not base_combos:
            return []

        configs = _accum_arith_configs()
        MAX_REG_COMBOS = 5
        capped_combos = base_combos[:MAX_REG_COMBOS]

        combos = []
        for cfg in configs:
            for base in capped_combos:
                combo = dict(base)
                combo["_vmac_config"] = cfg
                combos.append(combo)
        return combos

    def generate_test_point(self, instr, regs, in_offset, out_offset, **_kw):
        """Generate test point with config register initialization.

        Identical structure to VmacStrategy.generate_test_point -- load
        inputs, set config via mov, execute, store outputs.
        """
        lines = []
        name = instr["name"]
        config_val = regs.get("_vmac_config", 0)
        lines.append(f"  // ---- test (accum_arith): {name} config=0x{config_val:x} ----")

        op_by_name = {op["name"]: op for op in instr.get("operands", [])}
        outputs = detect_output_operands(instr)
        output_names = {op["name"] for op in outputs}
        asm_op_names = re.findall(r'\$(\w+)', instr.get("asm_string", ""))

        # Load input registers from [p0, #offset].
        cur_in_offset = in_offset
        for op_name in asm_op_names:
            if op_name in output_names:
                continue
            if op_name not in op_by_name:
                continue
            op = op_by_name[op_name]
            kind = _effective_kind(op)
            if not kind:
                continue
            if kind not in KNOWN_REGISTER_KINDS:
                continue
            if op_name == "c":
                continue  # Config set via mov, not loaded from buffer.
            align = 32 if kind in ("vector256", "vector512", "accumulator",
                                    "wide_y", "sparse_qx", "quad") else 4
            cur_in_offset = _align(cur_in_offset, align)
            reg = regs.get(op_name, "r0")
            load_lines = _load_instruction(reg, kind, "p0", cur_in_offset)
            lines.extend(load_lines)
            cur_in_offset += _operand_size(op, name)

        # NOP sled for load pipeline latency.
        lines.extend(_nop_sled(LOAD_LATENCY))

        # Initialize the config register with a valid config word.
        config_reg = regs.get("c", "r0")
        if config_val <= 0x1FF:
            lines.append(f"  mov {config_reg}, #{config_val}")
        else:
            lines.append(f"  movxm {config_reg}, #{config_val}")

        # The instruction itself.
        asm_line = "  " + _substitute_asm(instr["asm_string"], regs,
                                          has_modifier=_has_modifier_operand(instr))
        lines.append(asm_line)

        # NOP sled after instruction: must cover result latency.
        lines.extend(_nop_sled(result_latency(instr)))

        # Store output registers to [p1, #offset].
        cur_out_offset = out_offset
        for op in outputs:
            kind = _effective_kind(op) or op.get("register_kind", "")
            align = 32 if kind in ("vector256", "vector512", "accumulator",
                                    "wide_y", "sparse_qx", "quad") else 4
            cur_out_offset = _align(cur_out_offset, align)
            reg = regs.get(op["name"], "r0")
            store_lines = _store_instruction(reg, kind, "p1", cur_out_offset)
            lines.extend(store_lines)
            cur_out_offset += _operand_size(op, name)

        lines.append("")
        return "\n".join(lines)

    def compute_input_size(self, instr, regs):
        """Input size excludes the config register (set via mov)."""
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
            if op_name == "c":
                continue
            op = op_by_name[op_name]
            kind = _effective_kind(op)
            if not kind or kind not in KNOWN_REGISTER_KINDS:
                continue
            align = 32 if kind in ("vector256", "vector512", "accumulator",
                                    "wide_y", "sparse_qx", "quad") else 4
            total = _align(total, align)
            total += _operand_size(op, instr_name)
        return max(4, total)


class VmacStrategy(TestStrategy):
    """Tests MAC/MUL instructions with valid config words.

    MAC instructions (vmac, vmul, vaddmac, vsubmac, vnegmac, etc.) take a
    scalar config register ($c) that controls tile geometry, element types,
    signedness, and accumulate mode.  Without a valid config word, the
    hardware produces undefined results.

    This strategy:
    1. Detects MAC instructions by mnemonic.
    2. Generates combos with valid config words derived from the encoding name.
    3. Injects `mov rN, #config_word` before the instruction to set the
       config register to a known-valid value (overriding random test data).
    """

    def can_test(self, instr: dict) -> tuple[bool, str]:
        mnemonic = instr.get("mnemonic", "")
        if mnemonic not in _VMAC_MNEMONICS:
            return (False, "not a MAC instruction")

        # Delegate to ComputeStrategy for the base testability check
        # (operand types, output detection, etc.).
        if _is_sp_relative(instr):
            return (False, "SP-relative MAC (unexpected)")
        status, reason = classify_instruction(instr)
        if status != "testable":
            return (False, reason)

        return (True, "")

    def _find_config_operand(self, instr: dict) -> Optional[str]:
        """Find the config register operand name (usually 'c')."""
        for op in instr.get("operands", []):
            if op["name"] == "c" and op.get("register_kind") == "scalar":
                return "c"
        return None

    def generate_combos(self, instr: dict) -> list[dict[str, str]]:
        """Generate combos that pair register assignments with valid configs.

        For each valid config word, generate one combo with default registers.
        The config word is stored in the combo under key '_vmac_config' (an
        internal key that generate_test_point picks up for the mov init).
        """
        # Get base register combos (varying accumulator, vector, etc.).
        base_combos = generate_operand_combos(instr)
        if not base_combos:
            return []

        # Get valid config words for this instruction.
        configs = _vmac_configs_for_instr(instr["name"])
        if not configs:
            return base_combos  # Fallback: no special config handling.

        # Cross-product: each config word x each register combo.
        # Cap register combos to avoid explosion (4 configs * 19 reg combos = 76).
        MAX_REG_COMBOS = 5
        capped_combos = base_combos[:MAX_REG_COMBOS]

        combos = []
        for cfg in configs:
            for base in capped_combos:
                combo = dict(base)
                combo["_vmac_config"] = cfg
                combos.append(combo)

        return combos

    def generate_test_point(self, instr, regs, in_offset, out_offset, **_kw):
        """Generate test point with config register initialization.

        Injects a `mov rN, #config_word` sequence after input loads but
        before the instruction execution, overriding whatever random data
        was loaded into the config register.
        """
        lines = []
        name = instr["name"]
        config_val = regs.get("_vmac_config", 0)
        lines.append(f"  // ---- test (vmac): {name} config=0x{config_val:x} ----")

        # Build lookup: operand name -> operand dict.
        op_by_name = {op["name"]: op for op in instr.get("operands", [])}

        # Detect outputs.
        outputs = detect_output_operands(instr)
        output_names = {op["name"] for op in outputs}

        # Determine inputs from asm_string order.
        asm_op_names = re.findall(r'\$(\w+)', instr.get("asm_string", ""))

        # Load input registers from [p0, #offset].
        cur_in_offset = in_offset
        for op_name in asm_op_names:
            if op_name in output_names:
                continue
            if op_name not in op_by_name:
                continue
            op = op_by_name[op_name]
            kind = _effective_kind(op)
            if not kind:
                continue
            if kind not in KNOWN_REGISTER_KINDS:
                continue
            # Skip the config register -- we'll set it via mov.
            if op_name == "c":
                continue
            align = 32 if kind in ("vector256", "vector512", "accumulator",
                                    "wide_y", "sparse_qx", "quad") else 4
            cur_in_offset = _align(cur_in_offset, align)
            reg = regs.get(op_name, "r0")
            load_lines = _load_instruction(reg, kind, "p0", cur_in_offset)
            lines.extend(load_lines)
            cur_in_offset += _operand_size(op, name)

        # NOP sled for load pipeline latency.
        lines.extend(_nop_sled(LOAD_LATENCY))

        # Initialize the config register with a valid config word.
        # The config register is a scalar (typically r0), so we use movxm
        # to load the full 20-bit immediate (mov only supports 10-bit).
        config_reg = regs.get("c", "r0")
        if config_val <= 0x1FF:
            # Small enough for mov immediate (10-bit signed, but config
            # values are always positive and fit in 9 unsigned bits).
            lines.append(f"  mov {config_reg}, #{config_val}")
        else:
            # Use movxm for larger values (20-bit immediate).
            lines.append(f"  movxm {config_reg}, #{config_val}")

        # The instruction itself.
        asm_line = "  " + _substitute_asm(instr["asm_string"], regs,
                                          has_modifier=_has_modifier_operand(instr))
        lines.append(asm_line)

        # NOP sled after instruction: must cover result latency.
        lines.extend(_nop_sled(result_latency(instr)))

        # Store output registers to [p1, #offset].
        cur_out_offset = out_offset
        for op in outputs:
            kind = _effective_kind(op) or op.get("register_kind", "")
            align = 32 if kind in ("vector256", "vector512", "accumulator",
                                    "wide_y", "sparse_qx", "quad") else 4
            cur_out_offset = _align(cur_out_offset, align)
            reg = regs.get(op["name"], "r0")
            store_lines = _store_instruction(reg, kind, "p1", cur_out_offset)
            lines.extend(store_lines)
            cur_out_offset += _operand_size(op, name)

        lines.append("")
        return "\n".join(lines)

    def compute_input_size(self, instr, regs):
        """Input size excludes the config register (set via mov, not loaded)."""
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
            if op_name == "c":
                continue  # Config register not loaded from buffer.
            op = op_by_name[op_name]
            kind = _effective_kind(op)
            if not kind or kind not in KNOWN_REGISTER_KINDS:
                continue
            align = 32 if kind in ("vector256", "vector512", "accumulator",
                                    "wide_y", "sparse_qx", "quad") else 4
            total = _align(total, align)
            total += _operand_size(op, instr_name)
        return max(4, total)


# ---------------------------------------------------------------------------
# Strategy Dispatch
# ---------------------------------------------------------------------------

STRATEGIES: list[TestStrategy] = [
    BranchStrategy(),
    # LockStrategy disabled: lock acq/rel instructions interact with the
    # objectfifo DMA lock protocol.  Register-indirect variants use PRNG
    # data as lock IDs, which can steal locks from the DMA engine and
    # deadlock the tile.  Needs a dedicated harness without objectfifos.
    # LockStrategy(),
    FifoLoadStrategy(),
    CascadeReadStrategy(),   # must be before CascadeStrategy
    CascadeStrategy(),
    StreamStrategy(),       # must be before ComputeStrategy
    # DoneStrategy disabled: done halts the core before lock releases,
    # so the output DMA never triggers and the runtime sequence hangs.
    # Needs a dedicated batch with custom lock handling.
    # DoneStrategy(),
    # EventStrategy disabled: event instruction has no testable output
    # and the marker-based approach adds risk for minimal value.
    # EventStrategy(),
    PointerArithStrategy(),
    PaddaSpStrategy(),
    LoadStrategy(),
    StoreStrategy(),
    AccumArithStrategy(),   # must be before ComputeStrategy
    VmacStrategy(),         # must be before ComputeStrategy
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
    # Resolve misclassified operands before inspecting the kind.
    op = _resolve_operand(op)
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
        align = 32 if kind in ("vector256", "vector512", "accumulator",
                                "wide_y", "sparse_qx", "quad") else 4
        total = _align(total, align)
        total += _operand_size(op, instr_name)
    # VINSERT: implicit r29 index register loaded from input (4 bytes).
    if instr_name.startswith("VINSERT"):
        total = _align(total, 4)
        total += 4
    return max(4, total)


def _compute_output_size(instr: dict) -> int:
    """Compute total output buffer size for a test point."""
    outputs = detect_output_operands(instr)
    instr_name = instr.get("name", "")
    total = 0
    for op in outputs:
        # Use effective kind so testable composite kinds are sized correctly.
        kind = _effective_kind(op) or op.get("register_kind", "")
        align = 32 if kind in ("vector256", "vector512", "accumulator",
                                "wide_y", "sparse_qx", "quad") else 4
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
    """Full pipeline: classify, generate combos, batch, write .s/.ll + manifest.

    Handles two kinds of batches:
      - Assembly (.s): normal instructions assembled by llvm-mc.
      - LLVM IR (.ll): conversion instructions compiled by llc via intrinsics.

    Args:
        isa_json_path: Path to aie2-isa.json.
        out_dir: Directory to write batch files and manifest.json.

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
    #
    # Conversion instructions are collected separately: one representative
    # per base mnemonic goes into the conversion batch (LLVM IR), while
    # all variants count toward testable_count.
    conv_strategy = ConversionStrategy()
    testable_count = 0
    skipped_count = 0
    skip_reasons: dict[str, int] = {}
    # (instr, combo_idx, regs, strategy) -- assembly test points only.
    test_point_specs: list[tuple[dict, int, dict, TestStrategy]] = []
    # Track which base conversion mnemonics we have seen (one repr each).
    seen_conv_bases: set[str] = set()
    # Conversion test points: list of dicts for generate_conversion_ll().
    conv_test_points: list[dict] = []
    # All conversion instruction defs (for testable count).
    conv_instr_count = 0
    # Cascade read specs: (instr, combo_idx, regs, strategy) for cascade_pair
    # generation.  These are NOT bin-packed; each becomes a standalone batch.
    cascade_specs: list[tuple[dict, int, dict, CascadeReadStrategy]] = []
    # Stream specs: (instr, combo_idx, regs, strategy) for stream_pair
    # generation.  Like cascade, NOT bin-packed; each becomes a standalone batch.
    stream_specs: list[tuple[dict, int, dict, StreamStrategy]] = []

    # Track all ISA names covered by each conversion base mnemonic.
    conv_base_to_isa_names: dict[str, list[str]] = {}
    # Branch instructions tested via .ll (not assembly).
    _BRANCH_MNEMONICS = frozenset({"j", "jl", "jnz", "jz", "ret", "jnzd"})
    branch_instr_count = 0

    for instr in all_instrs:
        # Check conversion first (before normal strategy dispatch).
        if conv_strategy.can_handle(instr):
            conv_instr_count += 1
            mnemonic = instr.get("mnemonic", "")
            base = _conversion_base_mnemonic(mnemonic)
            conv_base_to_isa_names.setdefault(base, []).append(instr["name"])
            if base not in seen_conv_bases:
                seen_conv_bases.add(base)
                info = CONVERSION_INTRINSICS[base]
                conv_test_points.append({"mnemonic": base, **info})
            continue

        # Branch instructions are tested via .ll, not assembly.
        # Count them as testable but don't dispatch to assembly strategies.
        mnemonic = instr.get("mnemonic", "")
        if mnemonic in _BRANCH_MNEMONICS:
            branch_instr_count += 1
            continue

        strategy, reason = classify_with_strategies(instr)
        if strategy is None:
            skipped_count += 1
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue

        # Cascade read instructions are counted as testable but NOT
        # bin-packed into mega-program batches.  Each becomes a standalone
        # cascade_pair batch with producer + consumer .s files.
        if isinstance(strategy, CascadeReadStrategy):
            testable_count += 1
            combos = strategy.generate_combos(instr)
            for combo_idx, regs in enumerate(combos):
                regs["_combo_idx"] = combo_idx
                cascade_specs.append((instr, combo_idx, regs, strategy))
            continue

        # Stream instructions are counted as testable but NOT bin-packed
        # into mega-program batches.  Each becomes a standalone stream_pair
        # batch with producer + consumer .s files.
        if isinstance(strategy, StreamStrategy):
            testable_count += 1
            combos = strategy.generate_combos(instr)
            for combo_idx, regs in enumerate(combos):
                regs["_combo_idx"] = combo_idx
                stream_specs.append((instr, combo_idx, regs, strategy))
            continue

        testable_count += 1
        combos = strategy.generate_combos(instr)

        for combo_idx, regs in enumerate(combos):
            # Tag the combo index so BranchStrategy can select taken/not-taken.
            regs["_combo_idx"] = combo_idx
            test_point_specs.append((instr, combo_idx, regs, strategy))

    # Count conversion instructions as testable.
    testable_count += conv_instr_count

    # Second pass: mov.d* also get non-stream combos via ComputeStrategy.
    # StreamStrategy already claims them for stream-source combos (ss0);
    # this pass lets ComputeStrategy handle non-stream combos (e.g., mov.d1 r0, r1).
    compute = ComputeStrategy()
    for instr in all_instrs:
        mnemonic = instr.get("mnemonic", "")
        if re.match(r"mov\.d[1-6]", mnemonic):
            can, _ = compute.can_test(instr)
            if can:
                testable_count += 1
                combos = compute.generate_combos(instr)
                for combo_idx, regs in enumerate(combos):
                    regs["_combo_idx"] = combo_idx
                    test_point_specs.append((instr, combo_idx, regs, compute))

    # First pass: generate assembly for each test point at offset 0 to
    # measure its code size (bundle count).  This lets us bin-pack into
    # batches that fit within program memory.
    measured_specs = []  # (instr, combo_idx, regs, bundle_count, strategy)
    for instr, combo_idx, regs, strategy in test_point_specs:
        asm = strategy.generate_test_point(instr, regs, in_offset=0, out_offset=0)
        bundles = _count_bundles(asm)
        measured_specs.append((instr, combo_idx, regs, bundles, strategy))

    # Bin-pack assembly batches by measured code size.
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

        # Second pass: regenerate assembly with correct per-batch offsets.
        batch_in_offset = 0
        batch_out_offset = 0
        batch_asm = []
        batch_meta = []
        # Track cumulative instruction word count for branch targets.
        batch_iw_count = 0

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
                code_iw_offset=batch_iw_count,
            )
            batch_iw_count += _count_bundles(asm)
            batch_asm.append(asm)
            batch_meta.append({
                "instruction": instr["name"],
                "slot": instr.get("slot", ""),
                "combo_index": combo_idx,
                "operands": {k: v for k, v in regs.items()
                             if not k.startswith("_")},
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
        write_if_changed(filepath, program)

        batches.append({
            "batch_index": batch_idx,
            "filename": filename,
            "source_type": "assembly",
            "test_count": len(batch_asm),
            "in_size": batch_in_offset,
            "out_size": batch_out_offset,
            "tests": batch_meta,
        })

        batch_idx += 1
        i = end

    # Generate conversion batch (LLVM IR) if there are conversion test points.
    total_conv_test_points = 0
    if conv_test_points:
        ll_content = generate_conversion_ll(conv_test_points)
        ll_filename = f"batch_{batch_idx:03d}.ll"
        ll_filepath = os.path.join(out_dir, ll_filename)
        write_if_changed(ll_filepath, ll_content)

        # Build metadata for each conversion test point.
        conv_meta = []
        conv_in_offset = 0
        conv_out_offset = 0
        for tp in conv_test_points:
            mnemonic = tp["mnemonic"]
            in_bytes = tp["in_bytes"]
            out_bytes = tp["out_bytes"]
            # For UPS/SRS, the actual I/O sizes differ from the intrinsic
            # in_bytes/out_bytes because we round-trip through acc.
            if ".ups." in mnemonic:
                effective_in = in_bytes
                effective_out = in_bytes  # round-trip back to vec
            elif ".srs." in mnemonic:
                effective_in = out_bytes  # loaded as vec
                effective_out = out_bytes
            else:
                effective_in = in_bytes
                effective_out = out_bytes

            # Track all ISA-level instruction names covered by this
            # base mnemonic (includes 2D/3D and addressing variants).
            covered_names = conv_base_to_isa_names.get(mnemonic, [mnemonic])
            conv_meta.append({
                "instruction": mnemonic,
                "slot": "conversion",
                "combo_index": 0,
                "operands": {},
                "in_offset": conv_in_offset,
                "in_size": effective_in,
                "out_offset": conv_out_offset,
                "out_size": effective_out,
                "covers_isa_names": covered_names,
            })
            conv_in_offset += effective_in
            conv_out_offset += effective_out

        total_conv_test_points = len(conv_test_points)
        batches.append({
            "batch_index": batch_idx,
            "filename": ll_filename,
            "source_type": "llvm_ir",
            "test_count": total_conv_test_points,
            "in_size": conv_in_offset,
            "out_size": conv_out_offset,
            "tests": conv_meta,
        })
        batch_idx += 1

    # Generate branch batch (LLVM IR) -- branches can't use assembly due to
    # absolute address encoding in llvm-mc (no relocations for j/jl/jnz/jz).
    branch_ll, branch_meta, branch_in, branch_out = generate_branch_ll()
    branch_filename = f"batch_{batch_idx:03d}.ll"
    branch_filepath = os.path.join(out_dir, branch_filename)
    write_if_changed(branch_filepath, branch_ll)

    batches.append({
        "batch_index": batch_idx,
        "filename": branch_filename,
        "source_type": "llvm_ir",
        "test_count": len(branch_meta),
        "in_size": branch_in,
        "out_size": branch_out,
        "tests": branch_meta,
    })
    batch_idx += 1

    # Generate cascade_pair batches.  Each cascade read instruction becomes
    # a standalone batch with producer + consumer .s files.
    for instr, combo_idx, regs, strategy in cascade_specs:
        pair = strategy.generate_cascade_pair(instr, regs)

        prod_asm = pair["producer_asm"]
        cons_asm = pair["consumer_asm"]

        # Wrap each in a standalone mega-program.
        prod_program = build_mega_program([prod_asm])
        cons_program = build_mega_program([cons_asm])

        prod_filename = f"batch_{batch_idx:03d}_producer.s"
        cons_filename = f"batch_{batch_idx:03d}_consumer.s"

        write_if_changed(os.path.join(out_dir, prod_filename), prod_program)
        write_if_changed(os.path.join(out_dir, cons_filename), cons_program)

        batches.append({
            "batch_index": batch_idx,
            "filename": cons_filename,
            "source_type": "cascade_pair",
            "producer_filename": prod_filename,
            "consumer_filename": cons_filename,
            "producer_in_size": strategy.compute_producer_input_size(),
            "producer_out_size": strategy.compute_producer_output_size(),
            "consumer_in_size": 0,
            "consumer_out_size": strategy.compute_consumer_output_size(instr),
            "instruction": instr["name"],
            "slot": instr.get("slot", ""),
            "test_count": 1,
            "in_size": strategy.compute_producer_input_size(),
            "out_size": (strategy.compute_producer_output_size()
                         + strategy.compute_consumer_output_size(instr)),
            "tests": [{
                "instruction": instr["name"],
                "slot": instr.get("slot", ""),
                "combo_index": combo_idx,
                "operands": {k: v for k, v in regs.items()
                             if not k.startswith("_")},
                "in_offset": 0,
                "in_size": strategy.compute_producer_input_size(),
                "out_offset": 0,
                "out_size": (strategy.compute_producer_output_size()
                             + strategy.compute_consumer_output_size(instr)),
            }],
        })
        batch_idx += 1

    # Generate stream_pair batches.  Each stream instruction becomes a
    # standalone batch with producer + consumer .s files.
    for instr, combo_idx, regs, strategy in stream_specs:
        pair = strategy.generate_stream_pair(instr, regs)

        prod_asm = pair["producer_asm"]
        cons_asm = pair["consumer_asm"]

        # Wrap each in a standalone mega-program.
        prod_program = build_mega_program([prod_asm])
        cons_program = build_mega_program([cons_asm])

        prod_filename = f"batch_{batch_idx:03d}_producer.s"
        cons_filename = f"batch_{batch_idx:03d}_consumer.s"

        write_if_changed(os.path.join(out_dir, prod_filename), prod_program)
        write_if_changed(os.path.join(out_dir, cons_filename), cons_program)

        batches.append({
            "batch_index": batch_idx,
            "filename": cons_filename,
            "source_type": "stream_pair",
            "producer_filename": prod_filename,
            "consumer_filename": cons_filename,
            "producer_in_size": strategy.compute_producer_input_size(instr),
            "producer_out_size": strategy.compute_producer_output_size(instr),
            "consumer_in_size": 0,
            "consumer_out_size": strategy.compute_consumer_output_size(instr),
            "instruction": instr["name"],
            "slot": instr.get("slot", ""),
            "test_count": 1,
            "in_size": 0,
            "out_size": (strategy.compute_producer_output_size(instr)
                         + strategy.compute_consumer_output_size(instr)),
            "tests": [{
                "instruction": instr["name"],
                "slot": instr.get("slot", ""),
                "combo_index": combo_idx,
                "operands": {k: v for k, v in regs.items()
                             if not k.startswith("_")},
                "in_offset": 0,
                "in_size": 0,
                "out_offset": 0,
                "out_size": (strategy.compute_producer_output_size(instr)
                             + strategy.compute_consumer_output_size(instr)),
            }],
        })
        batch_idx += 1

    # Write manifest.json.
    # Count branch .ll test points from the branch batch.
    branch_test_count = len(branch_meta)
    total_test_points = (len(test_point_specs) + total_conv_test_points
                         + len(cascade_specs) + len(stream_specs)
                         + branch_test_count)
    # Conversion and branch instructions are all testable (tested via .ll).
    total_conv_isa_names = sum(len(v) for v in conv_base_to_isa_names.values())
    manifest = {
        "testable_instructions": testable_count + conv_instr_count + branch_instr_count,
        "skipped_instructions": skipped_count,
        "conversion_variants_covered": total_conv_isa_names,
        "total_test_points": total_test_points,
        "total_batches": len(batches),
        "skip_reasons": skip_reasons,
        "batches": batches,
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest_content = json.dumps(manifest, indent=2) + "\n"
    if not write_if_changed(manifest_path, manifest_content):
        # Content unchanged, but touch the mtime so the staleness check
        # in isa-test.sh knows we verified the output is current.
        os.utime(manifest_path)

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

        conv_strategy = ConversionStrategy()
        testable_count = 0
        skipped_count = 0
        skip_reasons: dict[str, int] = {}
        strategy_counts: dict[str, int] = {}
        testable_instrs: list[tuple[dict, str]] = []

        for slot, instrs in isa_data.items():
            for instr in instrs:
                # Check conversion first.
                if conv_strategy.can_handle(instr):
                    testable_count += 1
                    sname = "ConversionStrategy"
                    strategy_counts[sname] = strategy_counts.get(sname, 0) + 1
                    testable_instrs.append((instr, sname))
                    continue

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

    testable = manifest['testable_instructions']
    skipped = manifest['skipped_instructions']
    total_isa = testable + skipped
    conv_variants = manifest.get('conversion_variants_covered', 0)
    print(f"Testable instructions: {testable}/{total_isa} ({100*testable/total_isa:.1f}%)")
    if conv_variants:
        conv_bases = len(set(
            t['instruction'] for b in manifest['batches'] for t in b['tests']
            if t.get('slot') == 'conversion'
        ))
        print(f"  (includes {conv_variants} conversion variants via {conv_bases} .ll test points)")
    print(f"Skipped instructions:  {skipped}")
    print(f"Total test points:     {manifest['total_test_points']}")
    print(f"Batches:               {manifest['total_batches']}")
    print(f"Output:                {args.out_dir}")
    print()
    print("Skip reasons:")
    for reason, count in sorted(manifest["skip_reasons"].items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
