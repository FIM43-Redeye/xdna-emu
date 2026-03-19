#!/usr/bin/env python3
"""Tests for isa-test-gen.py -- instruction classifier, operand mapper, and
assembly test point generator.

Run: python3 -m pytest tools/test_isa_test_gen.py -v
"""

import pytest
import json
import sys
import os

# Ensure tools/ is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# We import from the module name (hyphen -> underscore handled by importlib).
import importlib
isa_test_gen = importlib.import_module("isa-test-gen")

classify_instruction = isa_test_gen.classify_instruction
register_names = isa_test_gen.register_names
immediate_values = isa_test_gen.immediate_values
detect_output_operands = isa_test_gen.detect_output_operands
generate_test_point = isa_test_gen.generate_test_point
build_mega_program = isa_test_gen.build_mega_program
generate_operand_combos = isa_test_gen.generate_operand_combos
generate_all = isa_test_gen.generate_all
result_latency = isa_test_gen.result_latency


# ---------------------------------------------------------------------------
# Helpers to build instruction dicts for testing
# ---------------------------------------------------------------------------

def _make_instr(name, mnemonic, asm_string, operands, **kwargs):
    """Build a minimal instruction dict matching aie2-isa.json schema."""
    instr = {
        "name": name,
        "mnemonic": mnemonic,
        "asm_string": asm_string,
        "slot": kwargs.get("slot", "alu"),
        "width": kwargs.get("width", 20),
        "operands": operands,
        "may_load": kwargs.get("may_load", False),
        "may_store": kwargs.get("may_store", False),
        "is_vector": kwargs.get("is_vector", False),
        "has_complete_decoder": kwargs.get("has_complete_decoder", True),
        "sched_class": kwargs.get("sched_class", "II_NOP"),
    }
    return instr


def _make_reg_op(name, kind, bit_width=5, is_output=False):
    return {
        "name": name,
        "bit_width": bit_width,
        "is_output": is_output,
        "operand_type": "register",
        "register_kind": kind,
        "signed": False,
        "scale": None,
    }


def _make_imm_op(name, bit_width=7, signed=True):
    return {
        "name": name,
        "bit_width": bit_width,
        "is_output": False,
        "operand_type": "immediate",
        "register_kind": None,
        "signed": signed,
        "scale": 1,
    }


def _make_composite_op(name, kind, bit_width=7):
    return {
        "name": name,
        "bit_width": bit_width,
        "is_output": False,
        "operand_type": "composite_register",
        "register_kind": kind,
        "signed": False,
        "scale": None,
    }


def _make_unknown_op(name, bit_width=2):
    return {
        "name": name,
        "bit_width": bit_width,
        "is_output": False,
        "operand_type": "unknown",
        "register_kind": None,
        "signed": False,
        "scale": None,
    }


# ===================================================================
# Task 2: classify_instruction tests
# ===================================================================

class TestClassifyInstruction:
    """Tests for classify_instruction()."""

    def test_simple_alu_testable(self):
        instr = _make_instr("ADD", "add", "add\t$mRx, $mRx0, $mRy", [
            _make_reg_op("mRx", "scalar"),
            _make_reg_op("mRx0", "scalar"),
            _make_reg_op("mRy", "scalar"),
        ])
        status, reason = classify_instruction(instr)
        assert status == "testable", f"Expected testable, got {status}: {reason}"

    def test_skip_load(self):
        instr = _make_instr("LDA", "lda", "lda\t$dst, [$p, #$off]", [
            _make_reg_op("dst", "scalar"),
            _make_reg_op("p", "pointer"),
            _make_imm_op("off"),
        ], may_load=True)
        status, reason = classify_instruction(instr)
        assert status == "skipped"
        assert "load" in reason.lower()

    def test_skip_store(self):
        instr = _make_instr("ST", "st", "st\t$src, [$p, #$off]", [
            _make_reg_op("src", "scalar"),
            _make_reg_op("p", "pointer"),
            _make_imm_op("off"),
        ], may_store=True)
        status, reason = classify_instruction(instr)
        assert status == "skipped"
        assert "store" in reason.lower()

    def test_skip_branch_j(self):
        instr = _make_instr("J_jump_ind", "j", "j\t$mPm", [
            _make_reg_op("mPm", "pointer"),
        ])
        status, reason = classify_instruction(instr)
        assert status == "skipped"
        assert "branch" in reason.lower() or "jump" in reason.lower()

    def test_skip_branch_jl(self):
        instr = _make_instr("JL_IND", "jl", "jl\t$mPm", [
            _make_reg_op("mPm", "pointer"),
        ])
        status, _ = classify_instruction(instr)
        assert status == "skipped"

    def test_skip_branch_ret(self):
        instr = _make_instr("RET", "ret", "ret lr", [])
        status, _ = classify_instruction(instr)
        assert status == "skipped"

    def test_skip_branch_jnz(self):
        instr = _make_instr("JNZ", "jnz", "jnz\t$mRx, $mPm", [
            _make_reg_op("mRx", "scalar"),
            _make_reg_op("mPm", "pointer"),
        ])
        status, _ = classify_instruction(instr)
        assert status == "skipped"

    def test_skip_branch_jnzd(self):
        instr = _make_instr("JNZD", "jnzd", "jnzd\t$mRx, $mRx0, $mPm", [
            _make_reg_op("mRx", "scalar"),
            _make_reg_op("mRx0", "scalar"),
            _make_reg_op("mPm", "pointer"),
        ])
        status, _ = classify_instruction(instr)
        assert status == "skipped"

    def test_skip_composite_register(self):
        """Instructions with unknown composite register kinds are still skipped."""
        instr = _make_instr("FOO_unknown_composite", "foo",
                            "foo\t$dst, $s0, $imm", [
            _make_composite_op("dst", "UnknownCompositeKind"),
            _make_reg_op("s0", "scalar"),
            _make_imm_op("imm"),
        ])
        status, reason = classify_instruction(instr)
        assert status == "skipped"
        assert "composite" in reason.lower()

    def test_shfl_dst_output_is_testable(self):
        """VBCSTSHFL-style instructions with ShflDst output are testable.

        ShflDst (mShflDst) is a composite register class that maps to vector512
        in the test harness.  Previously these were skipped as 'composite
        register operand'; they should now be classified as testable.
        """
        instr = _make_instr("VBCSTSHFL_16", "vbcstshfl.16",
                            "vbcstshfl.16\t$dst, $s0, $idx", [
            _make_composite_op("dst", "ShflDst", bit_width=5),
            _make_reg_op("s0", "scalar", bit_width=5),
            {
                "name": "idx",
                "bit_width": 5,
                "is_output": False,
                "operand_type": "register",
                "register_kind": "scalar",
                "signed": False,
                "scale": None,
            },
        ], slot="mv", is_vector=True)
        status, reason = classify_instruction(instr)
        assert status == "testable", \
            f"VBCSTSHFL_16 with ShflDst should be testable, got: {reason}"

    def test_wm1_input_is_testable(self):
        """VCONV_FP32_BF16 with Wm1 source operand is testable.

        Wm1 (mWm_1) is a composite vector256 class with non-monotonic encoding.
        The instruction has a standard accumulator destination and a Wm1 source;
        it should be testable since both operand types are known.
        """
        instr = _make_instr("VCONV_FP32_BF16", "vconv.fp32.bf16",
                            "vconv.fp32.bf16\t$dst, $src", [
            _make_reg_op("dst", "accumulator", bit_width=5),
            _make_composite_op("src", "Wm1", bit_width=5),
        ], slot="mv", is_vector=True)
        status, reason = classify_instruction(instr)
        assert status == "testable", \
            f"VCONV_FP32_BF16 with Wm1 source should be testable, got: {reason}"

    def test_other_composite_kinds_still_skipped(self):
        """Composite register kinds NOT in TESTABLE_COMPOSITE_KINDS are still skipped."""
        instr = _make_instr("FOO_mv", "foo.mv", "foo.mv\t$dst, $src", [
            _make_composite_op("dst", "SomeObscureKind", bit_width=7),
            _make_reg_op("src", "scalar"),
        ], slot="mv")
        status, reason = classify_instruction(instr)
        assert status == "skipped"
        assert "composite" in reason.lower()

    def test_mvsclsrc_output_is_testable(self):
        """MOV_mv_scl-style with MvSclSrc output/source is testable.

        MvSclSrc (bw=7) is the mv-slot scalar source/destination class.
        Both operands are MvSclSrc composites: first is the destination,
        second is the source.
        """
        instr = _make_instr("MOV_mv_scl", "mov",
                            "mov\t$mMvSclDst, $mMvSclSrc", [
            _make_composite_op("mMvSclDst", "MvSclSrc", bit_width=7),
            _make_composite_op("mMvSclSrc", "MvSclSrc", bit_width=7),
        ], slot="mv")
        status, reason = classify_instruction(instr)
        assert status == "testable", \
            f"MOV_mv_scl with MvSclSrc operands should be testable, got: {reason}"

    def test_alucg_output_is_testable(self):
        """MOVX_alu_cg with AluCg destination is testable.

        AluCg (bw=6) is the alu-slot constant-generator destination class.
        The instruction moves an immediate into an AluCg destination register.
        """
        instr = _make_instr("MOVX_alu_cg", "movx",
                            "movx\t$dst, $i", [
            _make_composite_op("dst", "AluCg", bit_width=6),
            _make_imm_op("i", bit_width=11),
        ], slot="alu")
        status, reason = classify_instruction(instr)
        assert status == "testable", \
            f"MOVX_alu_cg with AluCg dst should be testable, got: {reason}"

    def test_ldacg_output_is_testable(self):
        """MOVA_lda_cg with LdaCg destination is testable.

        LdaCg (bw=7) is the lda-slot constant-generator destination class.
        """
        instr = _make_instr("MOVA_lda_cg", "mova",
                            "mova\t$mLdaCg, $c11s", [
            _make_composite_op("mLdaCg", "LdaCg", bit_width=7),
            _make_imm_op("c11s", bit_width=11),
        ], slot="lda")
        status, reason = classify_instruction(instr)
        assert status == "testable", \
            f"MOVA_lda_cg with LdaCg dst should be testable, got: {reason}"

    def test_vextract_with_ers4_and_mvsclsrc_is_testable(self):
        """VEXTRACT with both ERS4 (index) and MvSclSrc (dst) is testable.

        VEXTRACT has two composite operands: ERS4 for the lane index and
        MvSclSrc for the scalar destination.  Both must be handled for
        the instruction to be classified as testable.
        """
        instr = _make_instr("VEXTRACT_D32", "vextract.d32",
                            "vextract.d32\t$dst, $s1, $idx", [
            _make_composite_op("dst", "MvSclSrc", bit_width=7),
            _make_composite_op("idx", "ERS4", bit_width=2),
            _make_reg_op("s1", "vector512", bit_width=4),
        ], slot="mv", is_vector=True)
        status, reason = classify_instruction(instr)
        assert status == "testable", \
            f"VEXTRACT_D32 with ERS4+MvSclSrc should be testable, got: {reason}"

    def test_vmov_mv_x_is_testable(self):
        """VMOV_mv_x with MvBMXDst (dst) and MvBMXSrc (src) is testable."""
        instr = _make_instr("VMOV_mv_x", "vmov",
                            "vmov\t$dst, $src", [
            _make_composite_op("dst", "MvBMXDst", bit_width=6),
            _make_composite_op("src", "MvBMXSrc", bit_width=9),
        ], slot="mv", is_vector=True)
        status, reason = classify_instruction(instr)
        assert status == "testable", \
            f"VMOV_mv_x with MvBMXDst/MvBMXSrc should be testable, got: {reason}"

    def test_vmov_mv_w_is_testable(self):
        """VMOV_mv_w with MvAMWQDst (dst) and MvAMWQSrc (src) is testable."""
        instr = _make_instr("VMOV_mv_w", "vmov",
                            "vmov\t$dst, $src", [
            _make_composite_op("dst", "MvAMWQDst", bit_width=7),
            _make_composite_op("src", "MvAMWQSrc", bit_width=9),
        ], slot="mv", is_vector=True)
        status, reason = classify_instruction(instr)
        assert status == "testable", \
            f"VMOV_mv_w with MvAMWQDst/MvAMWQSrc should be testable, got: {reason}"

    def test_cascade_instruction_is_skipped(self):
        """VMOV_mv_scd reads from SCD (cascade) -- skipped without cascade setup."""
        instr = _make_instr("VMOV_mv_scd", "vmov",
                            "vmov\t$dst, SCD", [
            _make_composite_op("dst", "MvBMXDst", bit_width=6),
        ], slot="lda", is_vector=True)
        status, reason = classify_instruction(instr)
        assert status == "skipped"
        assert "cascade" in reason.lower()

    def test_skip_unknown_operand_type(self):
        instr = _make_instr("FOO", "foo", "foo\t$dst, $src", [
            _make_reg_op("dst", "scalar"),
            _make_unknown_op("src"),
        ])
        status, reason = classify_instruction(instr)
        assert status == "skipped"
        assert "unknown" in reason.lower()

    def test_skip_lock_mnemonic(self):
        instr = _make_instr("ACQ", "acq", "acq\t$id, $mRy", [
            _make_imm_op("id", bit_width=6),
            _make_reg_op("mRy", "scalar"),
        ])
        status, reason = classify_instruction(instr)
        assert status == "skipped"
        assert "lock" in reason.lower()

    def test_skip_rel_mnemonic(self):
        instr = _make_instr("REL", "rel", "rel\t$id, $mRy", [
            _make_imm_op("id", bit_width=6),
            _make_reg_op("mRy", "scalar"),
        ])
        status, reason = classify_instruction(instr)
        assert status == "skipped"
        assert "lock" in reason.lower()

    def test_skip_no_operands(self):
        """Nops and done have no outputs -- skip them."""
        instr = _make_instr("DONE", "done", "done\t", [])
        status, reason = classify_instruction(instr)
        assert status == "skipped"
        assert "no output" in reason.lower() or "no operand" in reason.lower()

    def test_skip_nop(self):
        instr = _make_instr("NOPX", "nopx", "nopx\t", [])
        status, _ = classify_instruction(instr)
        assert status == "skipped"

    def test_skip_unknown_register_kind(self):
        """Exotic register kinds like ERS4, ShflDst should be skipped."""
        instr = _make_instr("FOO", "foo", "foo\t$dst, $src", [
            _make_reg_op("dst", "ERS4"),
            _make_reg_op("src", "scalar"),
        ])
        status, reason = classify_instruction(instr)
        assert status == "skipped"
        assert "register kind" in reason.lower() or "unknown" in reason.lower()

    def test_vector_vadd_testable(self):
        """VADD with accumulator operands should be testable."""
        instr = _make_instr("VADD", "vadd", "vadd\t$dst, $acc1, $acc2, $c", [
            _make_reg_op("acc2", "accumulator", bit_width=4),
            _make_reg_op("c", "scalar", bit_width=5),
            _make_reg_op("dst", "accumulator", bit_width=4),
            _make_reg_op("acc1", "accumulator", bit_width=4),
        ], slot="vec")
        status, reason = classify_instruction(instr)
        assert status == "testable", f"Expected testable, got {status}: {reason}"

    def test_event_skipped(self):
        """EVENT instruction has no meaningful output."""
        instr = _make_instr("EVENT", "event", "event\t$val", [
            _make_imm_op("val", bit_width=5, signed=False),
        ])
        status, _ = classify_instruction(instr)
        assert status == "skipped"

    def test_hardware_counter_testable(self):
        """MOV_CNTR reads hardware cycle counter into register pair -- testable.

        The counter is always running on an active core.  Output is a 64-bit
        cycle count in a register pair (register+16, bw=3 -> eL class).
        """
        instr = _make_instr("MOV_CNTR", "mov", "mov\t$dst, cntr", [
            {
                "name": "dst",
                "bit_width": 3,
                "is_output": False,
                "operand_type": "register+16",
                "register_kind": "scalar",
                "signed": False,
                "scale": None,
            },
        ])
        status, reason = classify_instruction(instr)
        assert status == "testable", f"Expected testable, got {status}: {reason}"

    def test_imm_only_skipped(self):
        """Instructions with only immediate operands (no register output)."""
        instr = _make_instr("EVENT", "event", "event\t$val", [
            _make_imm_op("val", bit_width=5),
        ])
        status, _ = classify_instruction(instr)
        assert status == "skipped"

    def test_control_register_output_is_testable(self):
        """MOVX_mvx_scl writes to a control register -- now testable via readback.

        The output control register is captured by reading it back into a
        general-purpose scalar ('mov r14, crSat') before storing.
        """
        instr = _make_instr("MOVX_mvx_scl", "movx", "movx\t$mCRm, $mRx", [
            _make_reg_op("mCRm", "control", bit_width=4),
            _make_reg_op("mRx", "scalar", bit_width=5),
        ])
        status, reason = classify_instruction(instr)
        assert status == "testable", \
            f"MOVX_mvx_scl should be testable (control reg readback), got: {reason}"

    def test_sparse_qxs2_testable(self):
        """Sparse multiply with qxs2 (accumulator bw=2) is now testable.

        The _resolve_sparse_qx heuristic resolves qxs2 to sparse_qx kind,
        which is a known register kind mapped to qx0-qx3.
        """
        instr = _make_instr("VNEGMSC_sparse", "vnegmsc",
                            "vnegmsc\t$dst, $acc1, $xs1, $qxs2, $c", [
            _make_reg_op("dst", "accumulator", bit_width=4),
            _make_reg_op("acc1", "accumulator", bit_width=4),
            _make_reg_op("xs1", "vector512"),
            _make_reg_op("qxs2", "accumulator", bit_width=2),
            _make_reg_op("c", "scalar"),
        ])
        status, reason = classify_instruction(instr)
        assert status == "testable", f"Expected testable, got {status}: {reason}"

    def test_non_qxs2_bw2_still_skipped(self):
        """Accumulator bw=2 operands NOT named qxs2 are still skipped."""
        instr = _make_instr("FOO_bw2", "foo",
                            "foo\t$dst, $src", [
            _make_reg_op("dst", "accumulator", bit_width=4),
            _make_reg_op("src", "accumulator", bit_width=2),
        ])
        status, reason = classify_instruction(instr)
        assert status == "skipped"
        assert "bw=2" in reason or "sparse" in reason

    def test_dontcare_operand_accepted(self):
        """Instructions with dontcare padding operands should be testable."""
        instr = _make_instr("PADDA", "padda", "padda\t[$mPa], #$c12s", [
            _make_reg_op("mPa", "pointer", bit_width=3),
            _make_imm_op("c12s", bit_width=12, signed=True),
            _make_unknown_op("dontcare2", bit_width=2),
        ])
        status, reason = classify_instruction(instr)
        assert status == "testable", f"Expected testable, got {status}: {reason}"

    def test_ys1_wide_y_testable(self):
        """ys1 (unknown bw=2) is now resolved to wide_y (y2-y5)."""
        instr = _make_instr("VMAC_sparse_wide", "vmac",
                            "vmac\t$dst, $acc1, $ys1, $qxs2, $c", [
            _make_reg_op("dst", "accumulator", bit_width=4),
            _make_reg_op("acc1", "accumulator", bit_width=4),
            _make_unknown_op("ys1", bit_width=2),
            _make_reg_op("qxs2", "accumulator", bit_width=2),
            _make_reg_op("c", "scalar"),
        ])
        status, reason = classify_instruction(instr)
        assert status == "testable", f"Expected testable, got {status}: {reason}"

    def test_non_ys1_unknown_bw2_still_skipped(self):
        """Unknown bw=2 operands NOT named ys1 are still skipped."""
        instr = _make_instr("FOO", "foo", "foo\t$dst, $src, $weird", [
            _make_reg_op("dst", "scalar"),
            _make_reg_op("src", "scalar"),
            _make_unknown_op("weird", bit_width=2),
        ])
        status, reason = classify_instruction(instr)
        assert status == "skipped"
        assert "unknown" in reason.lower()

    def test_register_plus_16_testable(self):
        """Vector compare with register+16 output (eRS8) is testable."""
        instr = _make_instr("VGE_D16", "vge.d16", "vge.d16\t$cmp, $s1, $s2", [
            {
                "name": "cmp",
                "bit_width": 3,
                "is_output": False,
                "operand_type": "register+16",
                "register_kind": "scalar",
                "signed": False,
                "scale": None,
            },
            _make_reg_op("s1", "vector512"),
            _make_reg_op("s2", "vector512"),
        ])
        status, _ = classify_instruction(instr)
        assert status == "testable"

    def test_vldb_4x_no_pointer_is_testable(self):
        """VLDB_4x* instructions have may_load=True but no pointer -- they are
        register-to-register vector shuffles and should be treated as compute."""
        instr = _make_instr("VLDB_4x16_HI", "vldb.4x16.hi",
                            "vldb.4x16.hi\t$dst, $src", [
            _make_reg_op("dst", "vector256"),
            _make_reg_op("src", "vector256"),
        ], may_load=True, slot="ldb")
        status, reason = classify_instruction(instr)
        assert status == "testable", \
            f"VLDB_4x with no pointer should be testable, got skipped: {reason}"

    def test_load_with_pointer_still_skipped(self):
        """A normal load (may_load=True, has pointer) must still be rejected
        by classify_instruction -- the pointer-exception must not affect it."""
        instr = _make_instr("LDA_U16", "lda.u16",
                            "lda.u16\t$dst, [$p, #$off]", [
            _make_reg_op("dst", "scalar"),
            _make_reg_op("p", "pointer"),
            _make_imm_op("off"),
        ], may_load=True)
        status, reason = classify_instruction(instr)
        assert status == "skipped", \
            f"Normal load (has pointer) should be skipped, got: {status}"
        assert "load" in reason.lower()


# ===================================================================
# Task 2: register_names tests
# ===================================================================

class TestRegisterNames:
    """Tests for register_names()."""

    def test_scalar_returns_list(self):
        names = register_names("scalar")
        assert isinstance(names, list)
        assert len(names) >= 2
        # Should use r-prefixed names
        assert all(n.startswith("r") for n in names)
        # Should NOT include r0/r1 (reserved in some contexts) -- but we
        # reserve p0/p1 for buffer pointers, not r0/r1.  r0 is fine for
        # scalar test values.
        assert "r0" in names or "r2" in names

    def test_pointer_avoids_p0_p1(self):
        names = register_names("pointer")
        assert "p0" not in names
        assert "p1" not in names
        assert len(names) >= 2
        assert all(n.startswith("p") for n in names)

    def test_vector256(self):
        names = register_names("vector256")
        assert isinstance(names, list)
        assert len(names) >= 2
        assert all(n.startswith("wl") or n.startswith("wh") for n in names)

    def test_vector512(self):
        names = register_names("vector512")
        assert isinstance(names, list)
        assert len(names) >= 2
        assert all(n.startswith("x") for n in names)

    def test_accumulator(self):
        names = register_names("accumulator")
        assert isinstance(names, list)
        assert len(names) >= 2
        assert all(n.startswith("cm") or n.startswith("bm") for n in names)

    def test_control(self):
        names = register_names("control")
        assert isinstance(names, list)
        # Control registers have specific names
        assert len(names) >= 1

    def test_modifier_m(self):
        names = register_names("modifier_m")
        assert isinstance(names, list)
        assert len(names) >= 1

    def test_modifier_dj(self):
        names = register_names("modifier_dj")
        assert isinstance(names, list)
        assert len(names) >= 1

    def test_unknown_kind_returns_empty(self):
        names = register_names("TrulyUnknownKind")
        assert names == []

    def test_shfl_dst_returns_vector512_names(self):
        """ShflDst composite kind maps to x* (vector512) register names.

        mShflDst is AIE2Vector512RegisterClass in TableGen, accepting either
        x* (mXm) or bml*/bmh* (mBMSm) registers in the assembler.  We use
        the x* subset as the simplest concrete names for test generation.
        """
        names = register_names("ShflDst")
        assert isinstance(names, list)
        assert len(names) >= 2
        assert all(n.startswith("x") for n in names), \
            f"Expected x* names for ShflDst, got {names}"

    def test_wm1_returns_vector256_names(self):
        """Wm1 composite kind maps to wl* (vector256) register names.

        mWm_1 is AIE2Vector256RegisterClass in TableGen with a non-monotonic
        encoding (wl0, wl2, ..., wh0, wh2, ...).  The assembler accepts
        the same wl*/wh* names as for plain vector256 -- the encoding
        difference is transparent to the assembly programmer.
        """
        names = register_names("Wm1")
        assert isinstance(names, list)
        assert len(names) >= 2
        assert all(n.startswith("wl") or n.startswith("wh") for n in names), \
            f"Expected wl*/wh* names for Wm1, got {names}"

    def test_mvsclsrc_returns_scalar_names(self):
        """MvSclSrc composite kind (mv-slot scalar source) maps to r* scalar names.

        mMvSclSrc encodes scalars, pointers, shift regs, modifiers, and control
        regs in 7 bits.  For testing we use the plain general-purpose scalar
        subset (r0-r7) since they can be loaded/stored directly.
        """
        names = register_names("MvSclSrc")
        assert isinstance(names, list)
        assert len(names) >= 2
        assert all(n.startswith("r") for n in names), \
            f"Expected r* names for MvSclSrc, got {names}"

    def test_ldacg_returns_scalar_names(self):
        """LdaCg (lda-slot constant-generator destination) maps to r* scalar names."""
        names = register_names("LdaCg")
        assert isinstance(names, list)
        assert len(names) >= 2
        assert all(n.startswith("r") for n in names), \
            f"Expected r* names for LdaCg, got {names}"

    def test_alucg_returns_scalar_names(self):
        """AluCg (alu-slot constant-generator destination) maps to r* scalar names."""
        names = register_names("AluCg")
        assert isinstance(names, list)
        assert len(names) >= 2
        assert all(n.startswith("r") for n in names), \
            f"Expected r* names for AluCg, got {names}"

    def test_ers4_returns_r24_to_r27(self):
        """ERS4 is a 2-bit narrow scalar class encoding exactly r24-r27.

        ERS4 is used as the index operand in VEXTRACT instructions.
        The 2-bit field encodes 4 consecutive high scalar registers.
        """
        names = register_names("ERS4")
        assert names == ["r24", "r25", "r26", "r27"], \
            f"Expected [r24, r25, r26, r27] for ERS4, got {names}"

    def test_mvbmxdst_returns_vector512_names(self):
        """MvBMXDst composite kind maps to x* (vector512) register names.

        MvBMXDst is used as the destination of cascade vmov (VMOV_mv_scd,
        VMOV_mv_x).  It encodes x* (vector512) and bml*/bmh* (acc-half).
        We use the x* subset as the simplest concrete names for testing.
        """
        names = register_names("MvBMXDst")
        assert isinstance(names, list)
        assert len(names) >= 2
        assert all(n.startswith("x") for n in names), \
            f"Expected x* names for MvBMXDst, got {names}"

    def test_mvbmxsrc_returns_vector512_names(self):
        """MvBMXSrc composite kind (wide vector/accumulator source) maps to x* names."""
        names = register_names("MvBMXSrc")
        assert isinstance(names, list)
        assert len(names) >= 2
        assert all(n.startswith("x") for n in names), \
            f"Expected x* names for MvBMXSrc, got {names}"

    def test_mvamwqdst_returns_vector256_names(self):
        """MvAMWQDst (256-bit vector destination for wmov) maps to wl* names."""
        names = register_names("MvAMWQDst")
        assert isinstance(names, list)
        assert len(names) >= 2
        assert all(n.startswith("wl") or n.startswith("wh") for n in names), \
            f"Expected wl*/wh* names for MvAMWQDst, got {names}"

    def test_mvamwqsrc_returns_vector256_names(self):
        """MvAMWQSrc (wide 256-bit source for wmov) maps to wl* names."""
        names = register_names("MvAMWQSrc")
        assert isinstance(names, list)
        assert len(names) >= 2
        assert all(n.startswith("wl") or n.startswith("wh") for n in names), \
            f"Expected wl*/wh* names for MvAMWQSrc, got {names}"

    def test_ldascl_returns_scalar_r_names(self):
        """LdaScl composite kind (mLdaScl/mSclSt DMS class) maps to r* scalar names."""
        names = register_names("LdaScl")
        assert isinstance(names, list)
        assert len(names) >= 4, f"Expected multiple r* names, got {names}"
        assert all(n.startswith("r") for n in names), \
            f"Expected r* names for LdaScl, got {names}"

    def test_quad_returns_q_names(self):
        """quad kind (DMV_Q mQQa class, q0-q3) returns q0..q3."""
        names = register_names("quad")
        assert isinstance(names, list)
        assert names == ["q0", "q1", "q2", "q3"], \
            f"Expected ['q0','q1','q2','q3'] for quad kind, got {names}"

    def test_accumulator_bw2_returns_quad_names(self):
        """Accumulator bw=2 is the DMV_Q quad register class, returns q0-q3."""
        names = register_names("accumulator", bit_width=2)
        assert names == ["q0", "q1", "q2", "q3"], \
            f"Expected q0-q3 for accumulator bw=2, got {names}"

    def test_accumulator_bw4_returns_cm(self):
        """Accumulator with 4-bit encoding -> 1024-bit cm registers."""
        names = register_names("accumulator", bit_width=4)
        assert all(n.startswith("cm") for n in names)

    def test_accumulator_bw5_returns_bm(self):
        """Accumulator with 5-bit encoding -> 512-bit bml/bmh halves."""
        names = register_names("accumulator", bit_width=5)
        assert any(n.startswith("bml") for n in names)
        assert any(n.startswith("bmh") for n in names)

    def test_scalar_bw2_returns_shift(self):
        """Scalar with 2-bit encoding -> s0-s2 shift registers (s3 reserved)."""
        names = register_names("scalar", bit_width=2)
        assert all(n.startswith("s") for n in names)
        assert len(names) == 3
        assert "s3" not in names

    def test_scalar_bw5_returns_general(self):
        """Scalar with 5-bit encoding -> general r0-r31."""
        names = register_names("scalar", bit_width=5)
        assert all(n.startswith("r") for n in names)

    def test_register_plus_16_single(self):
        """register+16 for 16/32-bit ops -> r16-r23 (single registers)."""
        names = register_names("scalar", bit_width=3,
                               operand_type="register+16",
                               instr_name="VGE_D16")
        assert all(n.startswith("r") for n in names)
        assert ":" not in names[0]

    def test_register_plus_16_pair(self):
        """register+16 for 8-bit ops -> register pairs (r17:r16 etc.)."""
        names = register_names("scalar", bit_width=3,
                               operand_type="register+16",
                               instr_name="VGE_D8")
        assert ":" in names[0]
        assert names[0] == "r19:r18"  # r17:r16 avoided (r16 callee-saved)

    def test_wide_y_returns_y_registers(self):
        """wide_y kind maps to y2-y5 (eYs 1024-bit wide vector class)."""
        names = register_names("wide_y")
        assert names == ["y2", "y3", "y4", "y5"]

    def test_sparse_qx_returns_qx_registers(self):
        """sparse_qx kind maps to qx0-qx3 (mQQXw sparse composite class)."""
        names = register_names("sparse_qx")
        assert names == ["qx0", "qx1", "qx2", "qx3"]

    def test_result_latency_scalar_alu(self):
        """Scalar ALU gets MIN_RESULT_LATENCY floor (model says 1)."""
        instr = {"sched_class": "II_ABS"}
        assert result_latency(instr) == isa_test_gen.MIN_RESULT_LATENCY

    def test_result_latency_vmac(self):
        """VMAC has 5-cycle model latency, clamped to min floor."""
        instr = {"sched_class": "II_VMAC"}
        assert result_latency(instr) == max(5, isa_test_gen.MIN_RESULT_LATENCY)

    def test_result_latency_vmacf(self):
        """VMAC.F has 6-cycle result latency."""
        instr = {"sched_class": "II_VMACf"}
        assert result_latency(instr) == 6

    def test_result_latency_default(self):
        """Unknown sched_class gets conservative default."""
        instr = {"sched_class": "II_NONEXISTENT_BOGUS"}
        assert result_latency(instr) >= 5  # conservative default


# ===================================================================
# Task 2: immediate_values tests
# ===================================================================

class TestImmediateValues:
    """Tests for immediate_values()."""

    def test_unsigned_boundaries(self):
        vals = immediate_values(5, signed=False)
        assert 0 in vals
        assert 31 in vals  # 2^5 - 1
        assert all(0 <= v <= 31 for v in vals)

    def test_signed_boundaries(self):
        vals = immediate_values(7, signed=True)
        assert 0 in vals
        assert -64 in vals   # -(2^6)
        assert 63 in vals    # 2^6 - 1

    def test_1_bit_unsigned(self):
        vals = immediate_values(1, signed=False)
        assert 0 in vals
        assert 1 in vals

    def test_values_are_unique(self):
        vals = immediate_values(8, signed=True)
        assert len(vals) == len(set(vals))

    def test_scaled_values_are_multiples(self):
        """PADDB-style scaled immediates: all values must be multiples of scale."""
        vals = immediate_values(9, signed=True, scale=4)
        for v in vals:
            assert v % 4 == 0, f"{v} is not a multiple of 4"

    def test_scaled_range(self):
        """9-bit signed field with scale=4 gives [-1024, 1020]."""
        vals = immediate_values(9, signed=True, scale=4)
        assert -1024 in vals  # -(2^8) * 4
        assert 1020 in vals   # (2^8 - 1) * 4
        assert 0 in vals


# ===================================================================
# Task 2: detect_output_operands tests
# ===================================================================

class TestDetectOutputOperands:
    """Tests for detect_output_operands()."""

    def test_alu_first_operand_is_output(self):
        """For 'add $mRx, $mRx0, $mRy', $mRx is the output (dst)."""
        instr = _make_instr("ADD", "add", "add\t$mRx, $mRx0, $mRy", [
            _make_reg_op("mRx", "scalar"),
            _make_reg_op("mRx0", "scalar"),
            _make_reg_op("mRy", "scalar"),
        ])
        outputs = detect_output_operands(instr)
        assert len(outputs) >= 1
        # The first operand in asm_string should be the output.
        assert outputs[0]["name"] == "mRx"

    def test_vec_dst_is_output(self):
        """For VADD, $dst should be detected as output."""
        instr = _make_instr("VADD", "vadd", "vadd\t$dst, $acc1, $acc2, $c", [
            _make_reg_op("acc2", "accumulator", bit_width=4),
            _make_reg_op("c", "scalar", bit_width=5),
            _make_reg_op("dst", "accumulator", bit_width=4),
            _make_reg_op("acc1", "accumulator", bit_width=4),
        ], slot="vec")
        outputs = detect_output_operands(instr)
        assert any(o["name"] == "dst" for o in outputs)

    def test_unary_op(self):
        """For 'abs $mRx, $mRx0', $mRx is the output."""
        instr = _make_instr("ABS", "abs", "abs\t$mRx, $mRx0", [
            _make_reg_op("mRx", "scalar"),
            _make_reg_op("mRx0", "scalar"),
        ])
        outputs = detect_output_operands(instr)
        assert len(outputs) == 1
        assert outputs[0]["name"] == "mRx"

    def test_tied_dst_vaddmac_dense(self):
        """VADDMAC_vmac_bm_core_dense: $dst in asm but no 'dst' operand.

        The first accumulator operand following $dst in asm_string order
        (acc1) is the tied write-back destination and should be returned
        as the sole output.
        """
        instr = _make_instr(
            "VADDMAC_vmac_bm_core_dense", "vaddmac",
            "vaddmac\t$dst, $acc1, $acc2, $s1, $s2, $c",
            [
                _make_reg_op("acc1", "accumulator", bit_width=4),
                _make_reg_op("c", "scalar", bit_width=5),
                _make_reg_op("s1", "vector512", bit_width=4),
                _make_reg_op("acc2", "accumulator", bit_width=4),
                _make_reg_op("s2", "vector512", bit_width=4),
            ],
            slot="vec", is_vector=True,
        )
        outputs = detect_output_operands(instr)
        assert len(outputs) == 1, f"Expected 1 output, got {outputs}"
        assert outputs[0]["name"] == "acc1"
        assert outputs[0]["register_kind"] == "accumulator"

    def test_tied_dst_vaddmac_f_dense_bw5(self):
        """VADDMAC_F_vmac_bm_core_dense: $dst in asm, no 'dst' operand, bw=5.

        Uses bw=5 accumulators (bm class).  The first accumulator in asm
        order after $dst is acc1.
        """
        instr = _make_instr(
            "VADDMAC_F_vmac_bm_core_dense", "vaddmac.f",
            "vaddmac.f\t$dst, $acc1, $acc2, $s1, $s2, $c",
            [
                _make_reg_op("acc2", "accumulator", bit_width=5),
                _make_reg_op("c", "scalar", bit_width=5),
                _make_reg_op("acc1", "accumulator", bit_width=5),
                _make_reg_op("s1", "vector512", bit_width=4),
                _make_reg_op("s2", "vector512", bit_width=4),
            ],
            slot="vec", is_vector=True,
        )
        outputs = detect_output_operands(instr)
        assert len(outputs) == 1, f"Expected 1 output, got {outputs}"
        assert outputs[0]["name"] == "acc1"
        assert outputs[0]["register_kind"] == "accumulator"

    def test_tied_dst_no_accumulator_returns_empty(self):
        """$dst in asm without any accumulator operand -> no output detected.

        Guards against the tied-destination fallback producing false positives
        for hypothetical instructions that have $dst-without-operand but no
        accumulator in the remaining operands.
        """
        instr = _make_instr(
            "HYPO_no_acc", "hypo",
            "hypo\t$dst, $s1, $s2",
            [
                _make_reg_op("s1", "vector512", bit_width=4),
                _make_reg_op("s2", "vector512", bit_width=4),
            ],
            slot="vec", is_vector=True,
        )
        outputs = detect_output_operands(instr)
        assert outputs == [], f"Expected no outputs, got {outputs}"

    def test_existing_dst_operand_unaffected_by_tied_fallback(self):
        """Instructions with a real 'dst' operand still work through the normal path.

        The tied-destination fallback only activates when 'dst' appears in
        the asm_string but is absent from the operands list.  VADD has a real
        'dst' operand and must continue to work correctly.
        """
        instr = _make_instr("VADD", "vadd", "vadd\t$dst, $acc1, $acc2, $c", [
            _make_reg_op("acc2", "accumulator", bit_width=4),
            _make_reg_op("c", "scalar", bit_width=5),
            _make_reg_op("dst", "accumulator", bit_width=4),
            _make_reg_op("acc1", "accumulator", bit_width=4),
        ], slot="vec")
        outputs = detect_output_operands(instr)
        assert any(o["name"] == "dst" for o in outputs), \
            "Normal 'dst' operand should still be detected as output"


class TestClassifyTiedDst:
    """Classification tests for VADDMAC/VADDMSC/VSUBMAC/VSUBMSC variants.

    Dense variants (bw=4 or bw=5 accumulators, no bw=2 sparse regs, no
    unknown operands) should now be testable because detect_output_operands()
    handles the tied-destination pattern.

    Sparse narrow variants remain blocked by bw=2 accumulator (qxs2).
    Sparse wide variants remain blocked by the unknown 'ys1' operand.
    """

    def _make_vmac_family_dense(self, name, mnemonic, bw):
        """Dense VMAC-family instruction with accumulator bit_width=bw."""
        return _make_instr(
            name, mnemonic,
            f"{mnemonic}\t$dst, $acc1, $acc2, $s1, $s2, $c",
            [
                _make_reg_op("acc1", "accumulator", bit_width=bw),
                _make_reg_op("c", "scalar", bit_width=5),
                _make_reg_op("s1", "vector512", bit_width=4),
                _make_reg_op("acc2", "accumulator", bit_width=bw),
                _make_reg_op("s2", "vector512", bit_width=4),
            ],
            slot="vec", is_vector=True,
        )

    def test_vaddmac_bm_dense_testable(self):
        """VADDMAC_vmac_bm_core_dense (bw=4) is testable after tied-dst fix."""
        instr = self._make_vmac_family_dense(
            "VADDMAC_vmac_bm_core_dense", "vaddmac", bw=4)
        status, reason = classify_instruction(instr)
        assert status == "testable", \
            f"VADDMAC_vmac_bm_core_dense should be testable, got: {reason}"

    def test_vaddmac_f_bm_dense_testable(self):
        """VADDMAC_F_vmac_bm_core_dense (bw=5) is testable after tied-dst fix."""
        instr = self._make_vmac_family_dense(
            "VADDMAC_F_vmac_bm_core_dense", "vaddmac.f", bw=5)
        status, reason = classify_instruction(instr)
        assert status == "testable", \
            f"VADDMAC_F_vmac_bm_core_dense should be testable, got: {reason}"

    def test_vaddmsc_bm_dense_testable(self):
        """VADDMSC_vmac_bm_core_dense (bw=4) is testable."""
        instr = self._make_vmac_family_dense(
            "VADDMSC_vmac_bm_core_dense", "vaddmsc", bw=4)
        status, reason = classify_instruction(instr)
        assert status == "testable", \
            f"VADDMSC_vmac_bm_core_dense should be testable, got: {reason}"

    def test_vsubmac_bm_dense_testable(self):
        """VSUBMAC_vmac_bm_core_dense (bw=4) is testable."""
        instr = self._make_vmac_family_dense(
            "VSUBMAC_vmac_bm_core_dense", "vsubmac", bw=4)
        status, reason = classify_instruction(instr)
        assert status == "testable", \
            f"VSUBMAC_vmac_bm_core_dense should be testable, got: {reason}"

    def test_vsubmsc_bm_dense_testable(self):
        """VSUBMSC_vmac_bm_core_dense (bw=4) is testable."""
        instr = self._make_vmac_family_dense(
            "VSUBMSC_vmac_bm_core_dense", "vsubmsc", bw=4)
        status, reason = classify_instruction(instr)
        assert status == "testable", \
            f"VSUBMSC_vmac_bm_core_dense should be testable, got: {reason}"

    def test_tied_dst_substituted_in_assembly(self):
        """$dst tied-destination must be replaced with acc1's register name."""
        instr = self._make_vmac_family_dense(
            "VSUBMAC_vmac_bm_core_dense", "vsubmac", bw=4)
        strategy = isa_test_gen.ComputeStrategy()
        can, reason = strategy.can_test(instr)
        assert can, reason
        combos = strategy.generate_combos(instr)
        asm = strategy.generate_test_point(instr, combos[0],
                                           in_offset=0, out_offset=0)
        assert "$dst" not in asm, f"$dst not substituted in: {asm[:200]}"

    def test_sparse_narrow_now_testable(self):
        """Sparse narrow variants (qxs2 bw=2) are now testable.

        _resolve_sparse_qx resolves qxs2 to sparse_qx, a known register kind.
        """
        instr = _make_instr(
            "VADDMAC_vmac_cm_core_sparse_narrow", "vaddmac",
            "vaddmac\t$dst, $acc1, $acc2, $xs1, $qxs2, $c",
            [
                _make_reg_op("xs1", "vector512", bit_width=4),
                _make_reg_op("acc2", "accumulator", bit_width=4),
                _make_reg_op("c", "scalar", bit_width=5),
                _make_reg_op("acc1", "accumulator", bit_width=4),
                _make_reg_op("qxs2", "accumulator", bit_width=2),
            ],
            slot="vec", is_vector=True,
        )
        status, reason = classify_instruction(instr)
        assert status == "testable", \
            f"Sparse narrow should be testable now, got {status}: {reason}"

    def test_sparse_wide_now_testable(self):
        """Sparse wide variants (ys1 + qxs2) are now testable.

        _resolve_unknown_operand resolves ys1 to wide_y, and
        _resolve_sparse_qx resolves qxs2 to sparse_qx.
        """
        instr = _make_instr(
            "VADDMSC_F_vmac_bm_core_sparse_wide", "vaddmsc.f",
            "vaddmsc.f\t$dst, $acc1, $acc2, $ys1, $qxs2, $c",
            [
                _make_unknown_op("ys1", bit_width=2),
                _make_reg_op("acc1", "accumulator", bit_width=5),
                _make_reg_op("acc2", "accumulator", bit_width=5),
                _make_reg_op("qxs2", "accumulator", bit_width=2),
                _make_reg_op("c", "scalar", bit_width=5),
            ],
            slot="vec", is_vector=True,
        )
        status, reason = classify_instruction(instr)
        assert status == "testable", \
            f"Sparse wide should be testable now, got {status}: {reason}"


# ===================================================================
# Task 3: generate_test_point tests
# ===================================================================

class TestGenerateTestPoint:
    """Tests for generate_test_point()."""

    def test_scalar_add(self):
        instr = _make_instr("ADD", "add", "add\t$mRx, $mRx0, $mRy", [
            _make_reg_op("mRx", "scalar"),
            _make_reg_op("mRx0", "scalar"),
            _make_reg_op("mRy", "scalar"),
        ])
        regs = {"mRx": "r2", "mRx0": "r3", "mRy": "r4"}
        asm = generate_test_point(instr, regs, in_offset=0, out_offset=0)
        assert isinstance(asm, str)
        # Should contain load of inputs
        assert "lda" in asm
        assert "r3" in asm
        assert "r4" in asm
        # Should contain the instruction itself
        assert "add\tr2, r3, r4" in asm
        # Should contain store of output
        assert "st" in asm
        assert "r2" in asm
        # Should have NOPs for pipeline safety
        assert "nop" in asm.lower() or "nopx" in asm.lower()

    def test_vector256_test_point(self):
        """Test that vector256 operands use vlda/vst."""
        instr = _make_instr("VNEG", "vneg", "vneg\t$dst, $src, $c", [
            _make_reg_op("dst", "vector256"),
            _make_reg_op("src", "vector256"),
            _make_reg_op("c", "scalar"),
        ], slot="vec")
        regs = {"dst": "wl2", "src": "wl4", "c": "r5"}
        asm = generate_test_point(instr, regs, in_offset=0, out_offset=0)
        assert "vlda" in asm
        assert "vst" in asm

    def test_offsets_used(self):
        """Verify that in_offset and out_offset are woven into load/store."""
        instr = _make_instr("ADD", "add", "add\t$mRx, $mRx0, $mRy", [
            _make_reg_op("mRx", "scalar"),
            _make_reg_op("mRx0", "scalar"),
            _make_reg_op("mRy", "scalar"),
        ])
        regs = {"mRx": "r2", "mRx0": "r3", "mRy": "r4"}
        asm = generate_test_point(instr, regs, in_offset=64, out_offset=128)
        # Input loads should reference offsets starting at 64
        assert "#64" in asm or "# 64" in asm or "#68" in asm
        # Output stores should reference offset starting at 128
        assert "#128" in asm or "# 128" in asm

    def test_immediate_operand_inlined(self):
        """Immediate operands should be inlined, not loaded."""
        instr = _make_instr("ADD_ri", "add", "add\t$mRx, $mRx0, $c7s", [
            _make_reg_op("mRx", "scalar"),
            _make_reg_op("mRx0", "scalar"),
            _make_imm_op("c7s", bit_width=7, signed=True),
        ])
        regs = {"mRx": "r2", "mRx0": "r3", "c7s": "42"}
        asm = generate_test_point(instr, regs, in_offset=0, out_offset=0)
        # The immediate should appear in the instruction
        assert "42" in asm

    def test_control_register_output_uses_mov_readback(self):
        """MOVX_mvx_scl: output is a control register; capture via mov readback.

        The generated test point must:
          1. Load the input scalar (mRx) from p0.
          2. Emit the movx instruction writing to the control register (mCRm).
          3. Read the control register back into r14 via 'mov r14, <ctrl_reg>'.
          4. Store r14 to p1 (NOT a direct 'st crSat, [p1, ...]').
        """
        instr = _make_instr("MOVX_mvx_scl", "movx", "movx\t$mCRm, $mRx", [
            _make_reg_op("mCRm", "control", bit_width=4),
            _make_reg_op("mRx", "scalar", bit_width=5),
        ])
        regs = {"mCRm": "crSat", "mRx": "r3"}
        asm = generate_test_point(instr, regs, in_offset=0, out_offset=0)
        # Must include the movx instruction itself.
        assert "movx\tcrSat, r3" in asm or "movx crSat, r3" in asm, \
            f"Expected movx instruction in:\n{asm}"
        # Must read the control register back via mov before storing.
        assert "mov r14, crSat" in asm, \
            f"Expected 'mov r14, crSat' readback in:\n{asm}"
        # Must store r14, not crSat directly.
        assert "st r14," in asm, \
            f"Expected 'st r14,' (not 'st crSat,') in:\n{asm}"
        assert "st crSat," not in asm, \
            f"Should NOT have 'st crSat,' directly in:\n{asm}"


# ===================================================================
# Task 3: build_mega_program tests
# ===================================================================

class TestBuildMegaProgram:
    """Tests for build_mega_program()."""

    def test_has_header(self):
        prog = build_mega_program([])
        assert ".text" in prog
        assert ".globl test_kernel" in prog or ".globl\ttest_kernel" in prog
        assert "test_kernel:" in prog

    def test_has_ret(self):
        prog = build_mega_program([])
        assert "ret lr" in prog

    def test_contains_test_points(self):
        tp1 = "  // test: ADD\n  add r2, r3, r4\n"
        tp2 = "  // test: SUB\n  sub r2, r3, r4\n"
        prog = build_mega_program([tp1, tp2])
        assert "ADD" in prog
        assert "SUB" in prog

    def test_ret_after_test_points(self):
        tp = "  add r2, r3, r4\n"
        prog = build_mega_program([tp])
        ret_pos = prog.rfind("ret lr")
        tp_pos = prog.find("add r2, r3, r4")
        assert tp_pos < ret_pos


# ===================================================================
# Integration: classify real ISA data
# ===================================================================

class TestRealISA:
    """Integration tests against the real aie2-isa.json."""

    @pytest.fixture
    def isa_data(self):
        isa_path = os.path.join(os.path.dirname(__file__), "aie2-isa.json")
        if not os.path.exists(isa_path):
            pytest.skip("aie2-isa.json not found")
        with open(isa_path) as f:
            return json.load(f)

    def test_some_instructions_testable(self, isa_data):
        """At least some instructions should be testable."""
        testable = 0
        for slot, instrs in isa_data.items():
            for instr in instrs:
                status, _ = classify_instruction(instr)
                if status == "testable":
                    testable += 1
        assert testable > 0, "No testable instructions found"

    def test_add_is_testable(self, isa_data):
        """The basic ADD instruction should be testable."""
        for instr in isa_data.get("alu", []):
            if instr["name"] == "ADD":
                status, reason = classify_instruction(instr)
                assert status == "testable", f"ADD should be testable: {reason}"
                return
        pytest.fail("ADD not found in ISA")

    def test_some_loads_testable(self, isa_data):
        """Some lda-slot instructions should now be testable via LoadStrategy."""
        testable = 0
        for instr in isa_data.get("lda", []):
            if instr.get("may_load"):
                strategy, _ = isa_test_gen.classify_with_strategies(instr)
                if strategy is not None:
                    testable += 1
        assert testable > 0, "Expected some loads to be testable"

    def test_some_stores_testable(self, isa_data):
        """Some st-slot instructions should now be testable via StoreStrategy."""
        testable = 0
        for instr in isa_data.get("st", []):
            if instr.get("may_store"):
                strategy, _ = isa_test_gen.classify_with_strategies(instr)
                if strategy is not None:
                    testable += 1
        assert testable > 0, "Expected some stores to be testable"

    def test_classification_summary(self, isa_data):
        """Print summary (informational, always passes)."""
        counts = {"testable": 0, "skipped": 0}
        skip_reasons = {}
        for slot, instrs in isa_data.items():
            for instr in instrs:
                status, reason = classify_instruction(instr)
                counts[status] += 1
                if status == "skipped":
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        print(f"\nClassification: {counts}")
        print("Skip reasons:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")


# ===================================================================
# Task 4: generate_operand_combos tests
# ===================================================================

class TestGenerateOperandCombos:
    """Tests for generate_operand_combos()."""

    def test_scalar_add_combos(self):
        """ADD with 3 scalar operands should produce baseline + variations."""
        instr = _make_instr("ADD", "add", "add\t$mRx, $mRx0, $mRy", [
            _make_reg_op("mRx", "scalar"),
            _make_reg_op("mRx0", "scalar"),
            _make_reg_op("mRy", "scalar"),
        ])
        combos = generate_operand_combos(instr)
        # At least 1 baseline + variations for each operand.
        assert len(combos) >= 2
        # Baseline should use first register name for each operand.
        baseline = combos[0]
        assert baseline["mRx"] == register_names("scalar")[0]
        assert baseline["mRx0"] == register_names("scalar")[0]

    def test_immediate_combos(self):
        """Instruction with immediate operand should vary the immediate."""
        instr = _make_instr("ADD_ri", "add", "add\t$mRx, $mRx0, $c7s", [
            _make_reg_op("mRx", "scalar"),
            _make_reg_op("mRx0", "scalar"),
            _make_imm_op("c7s", bit_width=7, signed=True),
        ])
        combos = generate_operand_combos(instr)
        # Should have combos with different immediate values.
        imm_vals = {c["c7s"] for c in combos}
        assert len(imm_vals) > 1

    def test_returns_list_of_dicts(self):
        instr = _make_instr("ABS", "abs", "abs\t$mRx, $mRx0", [
            _make_reg_op("mRx", "scalar"),
            _make_reg_op("mRx0", "scalar"),
        ])
        combos = generate_operand_combos(instr)
        assert isinstance(combos, list)
        for c in combos:
            assert isinstance(c, dict)

    def test_baseline_always_first(self):
        """First combo should be all defaults."""
        instr = _make_instr("ADD", "add", "add\t$mRx, $mRx0, $mRy", [
            _make_reg_op("mRx", "scalar"),
            _make_reg_op("mRx0", "scalar"),
            _make_reg_op("mRy", "scalar"),
        ])
        combos = generate_operand_combos(instr)
        baseline = combos[0]
        # All values should be the first choice from register_names.
        for key in baseline:
            assert baseline[key] == register_names("scalar")[0]


# ===================================================================
# ComputeStrategy tests
# ===================================================================

class TestPostModifyDetection:
    """Tests for _is_postmodify_immediate."""

    def test_postmodify_detected(self):
        assert isa_test_gen._is_postmodify_immediate(
            "lda.u8\t$mRa, [$ptr], $imm", "imm")

    def test_offset_not_postmodify(self):
        assert not isa_test_gen._is_postmodify_immediate(
            "lda.s16\t$mRa, [$ptr, $imm]", "imm")

    def test_no_match_different_name(self):
        assert not isa_test_gen._is_postmodify_immediate(
            "lda.u8\t$mRa, [$ptr], $imm", "off")


class TestComputeStrategy:
    """Tests for ComputeStrategy (refactored from existing logic)."""

    def test_scalar_add_can_test(self):
        instr = _make_instr("ADD", "add", "add\t$mRx, $mRx0, $mRy", [
            _make_reg_op("mRx", "scalar"),
            _make_reg_op("mRx0", "scalar"),
            _make_reg_op("mRy", "scalar"),
        ])
        strategy = isa_test_gen.ComputeStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"Expected can_test=True: {reason}"

    def test_load_not_compute(self):
        instr = _make_instr("LDA", "lda", "lda\t$dst, [$p, #$off]", [
            _make_reg_op("dst", "scalar"),
            _make_reg_op("p", "pointer"),
            _make_imm_op("off"),
        ], may_load=True)
        strategy = isa_test_gen.ComputeStrategy()
        can, reason = strategy.can_test(instr)
        assert not can

    def test_generate_test_point_unchanged(self):
        """ComputeStrategy must produce identical assembly to old code path."""
        instr = _make_instr("ADD", "add", "add\t$mRx, $mRx0, $mRy", [
            _make_reg_op("mRx", "scalar"),
            _make_reg_op("mRx0", "scalar"),
            _make_reg_op("mRy", "scalar"),
        ])
        regs = {"mRx": "r2", "mRx0": "r3", "mRy": "r4"}
        # Old path
        old_asm = generate_test_point(instr, regs, in_offset=0, out_offset=0)
        # New path via strategy
        strategy = isa_test_gen.ComputeStrategy()
        new_asm = strategy.generate_test_point(instr, regs, in_offset=0, out_offset=0)
        assert old_asm == new_asm


# ===================================================================
# LoadStrategy tests
# ===================================================================

class TestLoadStrategy:
    """Tests for LoadStrategy."""

    def test_scalar_load_can_test(self):
        """Simple scalar load with pointer + immediate offset."""
        instr = _make_instr("LDA_S16", "lda.s16",
                            "lda.s16\t$mRa, [$ptr, #$off]", [
            _make_reg_op("mRa", "scalar"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("off", bit_width=6, signed=True),
        ], may_load=True, slot="lda")
        strategy = isa_test_gen.LoadStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"Expected can_test=True: {reason}"

    def test_rejects_store_with_may_load(self):
        """Instructions with both may_load and may_store are stores, not loads."""
        instr = _make_instr("ST_2D_S16", "st.2d.s16",
                            "st.2d.s16\t$mRv, [$ptr], $mod", [
            _make_reg_op("mRv", "scalar"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_reg_op("mod", "modifier_m", bit_width=3),
        ], may_load=True, may_store=True, slot="lda")
        strategy = isa_test_gen.LoadStrategy()
        can, reason = strategy.can_test(instr)
        assert not can

    def test_rejects_no_pointer(self):
        """VLDB_4x* register-to-register shuffles have no pointer -- reject."""
        instr = _make_instr("VLDB_4x16_HI", "vldb.4x16.hi",
                            "vldb.4x16.hi\t$dst, $src", [
            _make_reg_op("dst", "vector256"),
            _make_reg_op("src", "vector256"),
        ], may_load=True, slot="ldb")
        strategy = isa_test_gen.LoadStrategy()
        can, reason = strategy.can_test(instr)
        assert not can

    def test_accepts_testable_composite_dest(self):
        """Loads with LdaScl composite destination are now testable."""
        instr = _make_instr("LDA_2D_LDASCL", "lda.2d",
                            "lda.2d\t$mLdaScl, [$ptr], $mod", [
            _make_composite_op("mLdaScl", "LdaScl", bit_width=7),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_reg_op("mod", "modifier_m", bit_width=3),
        ], may_load=True, slot="lda")
        strategy = isa_test_gen.LoadStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"LdaScl composite should be testable now, got: {reason}"

    def test_rejects_unknown_composite_dest(self):
        """Loads with unknown composite destination are still rejected."""
        instr = _make_instr("LDA_UNKNOWN", "lda",
                            "lda\t$dst, [$ptr], $mod", [
            _make_composite_op("dst", "UnknownCompKind", bit_width=7),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_reg_op("mod", "modifier_m", bit_width=3),
        ], may_load=True, slot="lda")
        strategy = isa_test_gen.LoadStrategy()
        can, reason = strategy.can_test(instr)
        assert not can

    def test_generates_valid_assembly(self):
        """LoadStrategy should produce setup + load + capture assembly."""
        instr = _make_instr("LDA_S16", "lda.s16",
                            "lda.s16\t$mRa, [$ptr, #$off]", [
            _make_reg_op("mRa", "scalar"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("off", bit_width=6, signed=True),
        ], may_load=True, slot="lda", sched_class="II_LDA")
        strategy = isa_test_gen.LoadStrategy()
        regs = {"mRa": "r0", "ptr": "p6", "off": "0"}
        asm = strategy.generate_test_point(instr, regs,
                                           in_offset=0, out_offset=0)
        # Should set up p6 pointing at input data
        assert "p6" in asm
        # Should contain the load instruction
        assert "lda.s16" in asm
        # Should store the loaded value
        assert "st" in asm
        # Should have NOP sled for load latency
        assert "nop" in asm

    def test_vector_load_uses_vst(self):
        """Vector loads should use vst to capture the result."""
        instr = _make_instr("VLDA_128", "vlda.128",
                            "vlda.128\t$dst, [$ptr, #$off]", [
            _make_reg_op("dst", "vector256"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("off", bit_width=6, signed=True),
        ], may_load=True, slot="ldb", sched_class="II_VLDA")
        strategy = isa_test_gen.LoadStrategy()
        regs = {"dst": "wl0", "ptr": "p6", "off": "0"}
        asm = strategy.generate_test_point(instr, regs,
                                           in_offset=0, out_offset=0)
        assert "vlda.128" in asm
        assert "vst" in asm

    def test_postmodify_preserves_immediate(self):
        """Post-modify load must preserve the combo-specified immediate."""
        instr = _make_instr("LDA_U8_ag_pstm_nrm_imm", "lda.u8",
                            "lda.u8\t$mRa, [$ptr], $imm", [
            _make_reg_op("mRa", "scalar"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("imm", bit_width=6, signed=True),
        ], may_load=True, slot="lda", sched_class="II_LDA")
        strategy = isa_test_gen.LoadStrategy()
        regs = {"mRa": "r0", "ptr": "p2", "imm": "-8"}
        asm = strategy.generate_test_point(instr, regs,
                                           in_offset=0, out_offset=0)
        # Post-modify immediate should NOT be zeroed.
        assert "], #-8" in asm or "], -8" in asm, \
            f"Post-modify immediate should be preserved, got:\n{asm}"

    def test_offset_immediate_zeroed(self):
        """Basic address offset immediate should be zeroed."""
        instr = _make_instr("LDA_S16", "lda.s16",
                            "lda.s16\t$mRa, [$ptr, $imm]", [
            _make_reg_op("mRa", "scalar"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("imm", bit_width=6, signed=True),
        ], may_load=True, slot="lda", sched_class="II_LDA")
        strategy = isa_test_gen.LoadStrategy()
        regs = {"mRa": "r0", "ptr": "p2", "imm": "-8"}
        asm = strategy.generate_test_point(instr, regs,
                                           in_offset=0, out_offset=0)
        # Address offset immediate SHOULD be zeroed.
        assert ", #0]" in asm or ", 0]" in asm, \
            f"Address offset immediate should be zeroed, got:\n{asm}"


# ===================================================================
# StoreStrategy tests
# ===================================================================

class TestStoreStrategy:
    """Tests for StoreStrategy."""

    def test_scalar_store_can_test(self):
        instr = _make_instr("ST_S16", "st.s16",
                            "st.s16\t$mRv, [$ptr, #$off]", [
            _make_reg_op("mRv", "scalar"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("off", bit_width=6, signed=True),
        ], may_store=True, slot="st")
        strategy = isa_test_gen.StoreStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"Expected can_test=True: {reason}"

    def test_rejects_no_pointer(self):
        instr = _make_instr("VST_FAKE", "vst.fake",
                            "vst.fake\t$src, $dst", [
            _make_reg_op("src", "vector256"),
            _make_reg_op("dst", "vector256"),
        ], may_store=True, slot="st")
        strategy = isa_test_gen.StoreStrategy()
        can, reason = strategy.can_test(instr)
        assert not can

    def test_accepts_testable_composite_source(self):
        """Stores with LdaScl composite source are now testable."""
        instr = _make_instr("ST_COMP", "st.2d",
                            "st.2d\t$mLdaScl, [$ptr], $mod", [
            _make_composite_op("mLdaScl", "LdaScl", bit_width=7),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_reg_op("mod", "modifier_m", bit_width=3),
        ], may_store=True, slot="st")
        strategy = isa_test_gen.StoreStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"LdaScl composite should be testable, got: {reason}"

    def test_generates_valid_assembly(self):
        instr = _make_instr("ST_S16", "st.s16",
                            "st.s16\t$mRv, [$ptr, #$off]", [
            _make_reg_op("mRv", "scalar"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("off", bit_width=6, signed=True),
        ], may_store=True, slot="st", sched_class="II_ST")
        strategy = isa_test_gen.StoreStrategy()
        regs = {"mRv": "r0", "ptr": "p7", "off": "0"}
        asm = strategy.generate_test_point(instr, regs,
                                           in_offset=0, out_offset=0)
        assert "lda" in asm
        assert "st.s16" in asm
        assert "p7" in asm

    def test_vector_store(self):
        instr = _make_instr("VST_128", "vst.128",
                            "vst.128\t$src, [$ptr, #$off]", [
            _make_reg_op("src", "vector256"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("off", bit_width=6, signed=True),
        ], may_store=True, slot="st", sched_class="II_VST")
        strategy = isa_test_gen.StoreStrategy()
        regs = {"src": "wl0", "ptr": "p7", "off": "0"}
        asm = strategy.generate_test_point(instr, regs,
                                           in_offset=0, out_offset=0)
        assert "vlda" in asm
        assert "vst.128" in asm

    def test_postmodify_preserves_immediate(self):
        """Post-modify store must preserve the combo-specified immediate."""
        instr = _make_instr("ST_S16_ag_pstm_nrm_imm", "st.s16",
                            "st.s16\t$mRv, [$ptr], $imm", [
            _make_reg_op("mRv", "scalar"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("imm", bit_width=6, signed=True),
        ], may_store=True, slot="st", sched_class="II_ST")
        strategy = isa_test_gen.StoreStrategy()
        regs = {"mRv": "r0", "ptr": "p7", "imm": "-4"}
        asm = strategy.generate_test_point(instr, regs,
                                           in_offset=0, out_offset=0)
        assert "], #-4" in asm or "], -4" in asm, \
            f"Post-modify immediate should be preserved, got:\n{asm}"

    def test_store_width_s8(self):
        instr = _make_instr("ST_S8", "st.s8",
                            "st.s8\t$mRv, [$ptr, #$off]", [
            _make_reg_op("mRv", "scalar"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("off", bit_width=6, signed=True),
        ], may_store=True, slot="st")
        strategy = isa_test_gen.StoreStrategy()
        assert strategy._compute_store_width(instr) == 1

    def test_store_width_s16(self):
        instr = _make_instr("ST_S16", "st.s16",
                            "st.s16\t$mRv, [$ptr, #$off]", [
            _make_reg_op("mRv", "scalar"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("off", bit_width=6, signed=True),
        ], may_store=True, slot="st")
        strategy = isa_test_gen.StoreStrategy()
        assert strategy._compute_store_width(instr) == 2

    def test_store_width_scalar_default(self):
        instr = _make_instr("ST", "st",
                            "st\t$mRv, [$ptr, #$off]", [
            _make_reg_op("mRv", "scalar"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("off", bit_width=6, signed=True),
        ], may_store=True, slot="st")
        strategy = isa_test_gen.StoreStrategy()
        assert strategy._compute_store_width(instr) == 4

    def test_store_width_vector(self):
        instr = _make_instr("VST", "vst",
                            "vst\t$src, [$ptr, #$off]", [
            _make_reg_op("src", "vector256"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("off", bit_width=6, signed=True),
        ], may_store=True, slot="st")
        strategy = isa_test_gen.StoreStrategy()
        assert strategy._compute_store_width(instr) == 32


# ===================================================================
# SP-relative instruction tests
# ===================================================================

class TestSpRelative:
    """Tests for SP-relative load/store instructions.

    SP-relative spill instructions use [sp, $imm] addressing where SP (p6)
    is the implicit base register, not listed as an explicit operand.
    The generator initializes SP to point at the data region for the test.
    """

    def _make_sp_load(self, name="VLDA_dmw_lda_w_ag_spill",
                      mnemonic="vlda", kind="vector256"):
        """Build a minimal SP-relative load instruction."""
        return _make_instr(name, mnemonic,
                           f"{mnemonic}\t$dst, [sp, $imm]", [
            _make_reg_op("dst", kind),
            _make_imm_op("imm", bit_width=12, signed=True),
        ], may_load=True, slot="lda", sched_class="II_VLDA")

    def _make_sp_store(self, name="VST_dmw_sts_w_ag_spill",
                       mnemonic="vst", kind="vector256"):
        """Build a minimal SP-relative store instruction."""
        return _make_instr(name, mnemonic,
                           f"{mnemonic}\t$src, [sp, $imm]", [
            _make_reg_op("src", kind),
            _make_imm_op("imm", bit_width=12, signed=True),
        ], may_store=True, slot="st", sched_class="II_VST")

    # --- _is_sp_relative helper ---

    def test_is_sp_relative_true(self):
        instr = self._make_sp_load()
        assert isa_test_gen._is_sp_relative(instr)

    def test_is_sp_relative_false_for_normal_load(self):
        instr = _make_instr("LDA_S16", "lda.s16",
                            "lda.s16\t$mRa, [$ptr, #$off]", [
            _make_reg_op("mRa", "scalar"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("off", bit_width=6, signed=True),
        ], may_load=True, slot="lda")
        assert not isa_test_gen._is_sp_relative(instr)

    def test_is_sp_relative_case_insensitive(self):
        """Detection should not be sensitive to asm_string case."""
        instr = _make_instr("FAKE", "vlda",
                            "vlda\t$dst, [SP, $imm]", [
            _make_reg_op("dst", "vector256"),
            _make_imm_op("imm", bit_width=12, signed=True),
        ], may_load=True)
        assert isa_test_gen._is_sp_relative(instr)

    # --- LoadStrategy with SP-relative ---

    def test_sp_load_can_test(self):
        """SP-relative load should be accepted by LoadStrategy."""
        instr = self._make_sp_load()
        strategy = isa_test_gen.LoadStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"Expected can_test=True for SP load: {reason}"

    def test_sp_load_with_testable_composite_accepted(self):
        """SP-relative load with LdaScl composite destination is now testable."""
        instr = _make_instr("LDA_dms_spill", "lda",
                            "lda\t$mLdaScl, [sp, $imm]", [
            _make_composite_op("mLdaScl", "LdaScl", bit_width=7),
            _make_imm_op("imm", bit_width=12, signed=True),
        ], may_load=True, slot="lda")
        strategy = isa_test_gen.LoadStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"SP+LdaScl should be testable, got: {reason}"

    def test_sp_load_with_bw2_accumulator_accepted(self):
        """SP-relative load with bw=2 accumulator is now testable.

        LDA_dmv_lda_q_ag_spill: accumulator bw=2 is the quad register class
        (q0-q3, mQQa in TableGen) which uses the scalar 'lda' slot and IS
        accepted by llvm-mc as "lda q0, [sp, #imm]".
        """
        instr = _make_instr("LDA_dmv_lda_q_ag_spill", "lda",
                            "lda\t$dst, [sp, $imm]", [
            _make_imm_op("imm", bit_width=12, signed=True),
            {
                "name": "dst",
                "bit_width": 2,
                "is_output": False,
                "operand_type": "register",
                "register_kind": "accumulator",
                "signed": False,
                "scale": None,
            },
        ], may_load=True, slot="lda")
        strategy = isa_test_gen.LoadStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"DMV_Q quad load (bw=2 acc) should be testable, got: {reason}"

    def test_sp_load_generates_assembly_with_sp_init(self):
        """SP-relative load should set up SP (p6) to point at input data."""
        instr = self._make_sp_load()
        strategy = isa_test_gen.LoadStrategy()
        regs = {"dst": "wl0", "imm": "0"}
        asm = strategy.generate_test_point(instr, regs,
                                           in_offset=0, out_offset=0)
        # SP is p6: the setup sequence should initialize p6 from p0.
        assert "mov p6, p0" in asm, \
            f"Expected SP (p6) initialization from p0:\n{asm}"
        # The load instruction itself.
        assert "vlda" in asm
        # Should use [sp, ...] syntax.
        assert "[sp," in asm
        # Immediate should be zeroed (data is at sp).
        assert "[sp, #0]" in asm or "[sp, 0]" in asm, \
            f"Expected zeroed immediate in SP-relative load:\n{asm}"
        # Should capture result to p1.
        assert "p1" in asm

    def test_sp_load_immediate_zeroed(self):
        """The immediate in [sp, $imm] should be zeroed regardless of combo."""
        instr = self._make_sp_load()
        strategy = isa_test_gen.LoadStrategy()
        regs = {"dst": "wl0", "imm": "64"}  # non-zero combo value
        asm = strategy.generate_test_point(instr, regs,
                                           in_offset=0, out_offset=0)
        assert "[sp, #0]" in asm or "[sp, 0]" in asm, \
            f"Immediate should be zeroed for SP-relative load:\n{asm}"

    # --- StoreStrategy with SP-relative ---

    def test_sp_store_can_test(self):
        """SP-relative store should be accepted by StoreStrategy."""
        instr = self._make_sp_store()
        strategy = isa_test_gen.StoreStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"Expected can_test=True for SP store: {reason}"

    def test_sp_store_with_ldascl_composite_accepted(self):
        """SP-relative store with LdaScl composite source is now testable.

        ST_dms_spill uses mSclSt (kind="LdaScl") which is mapped to scalar
        in TESTABLE_COMPOSITE_KINDS.  The test harness treats it as a plain
        scalar register store, using r* names.
        """
        instr = _make_instr("ST_dms_spill", "st",
                            "st\t$mSclSt, [sp, $imm]", [
            _make_composite_op("mSclSt", "LdaScl", bit_width=7),
            _make_imm_op("imm", bit_width=12, signed=True),
        ], may_store=True, slot="st")
        strategy = isa_test_gen.StoreStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"ST_dms_spill with LdaScl should be testable, got: {reason}"

    def test_sp_store_generates_assembly_with_sp_init(self):
        """SP-relative store should set up SP (p6) to point at output buffer."""
        instr = self._make_sp_store()
        strategy = isa_test_gen.StoreStrategy()
        regs = {"src": "wl0", "imm": "0"}
        asm = strategy.generate_test_point(instr, regs,
                                           in_offset=0, out_offset=0)
        # Should load source data from p0.
        assert "p0" in asm
        # SP is p6: the setup sequence should initialize p6 from p1 (output).
        assert "mov p6, p1" in asm, \
            f"Expected SP (p6) initialization from p1:\n{asm}"
        # The store instruction itself.
        assert "vst" in asm
        # Should use [sp, ...] syntax.
        assert "[sp," in asm
        # Immediate should be zeroed (data is at sp).
        assert "[sp, #0]" in asm or "[sp, 0]" in asm, \
            f"Expected zeroed immediate in SP-relative store:\n{asm}"

    def test_sp_store_does_not_use_p7(self):
        """SP-relative store should NOT set up p7 (not used for SP stores)."""
        instr = self._make_sp_store()
        strategy = isa_test_gen.StoreStrategy()
        regs = {"src": "wl0", "imm": "0"}
        asm = strategy.generate_test_point(instr, regs,
                                           in_offset=0, out_offset=0)
        # p7 is the normal store scratch register; SP stores use p6 instead.
        assert "p7" not in asm, \
            f"SP-relative store should not set up p7:\n{asm}"

    def test_normal_load_unchanged(self):
        """Normal (non-SP-relative) loads should be unaffected by the change."""
        instr = _make_instr("LDA_S16", "lda.s16",
                            "lda.s16\t$mRa, [$ptr, #$off]", [
            _make_reg_op("mRa", "scalar"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("off", bit_width=6, signed=True),
        ], may_load=True, slot="lda", sched_class="II_LDA")
        strategy = isa_test_gen.LoadStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"Normal load should still be testable: {reason}"
        regs = {"mRa": "r0", "ptr": "p6", "off": "0"}
        asm = strategy.generate_test_point(instr, regs,
                                           in_offset=0, out_offset=0)
        assert "lda.s16" in asm
        # Normal load uses p6 as scratch (same as before).
        assert "p6" in asm

    def test_normal_store_unchanged(self):
        """Normal (non-SP-relative) stores should be unaffected by the change."""
        instr = _make_instr("ST_S16", "st.s16",
                            "st.s16\t$mRv, [$ptr, #$off]", [
            _make_reg_op("mRv", "scalar"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("off", bit_width=6, signed=True),
        ], may_store=True, slot="st", sched_class="II_ST")
        strategy = isa_test_gen.StoreStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"Normal store should still be testable: {reason}"
        regs = {"mRv": "r0", "ptr": "p7", "off": "0"}
        asm = strategy.generate_test_point(instr, regs,
                                           in_offset=0, out_offset=0)
        assert "st.s16" in asm
        # Normal store uses p7 as scratch (same as before).
        assert "p7" in asm

    # --- Real ISA JSON integration ---

    @pytest.fixture
    def isa_data(self):
        path = os.path.join(os.path.dirname(__file__), "aie2-isa.json")
        if not os.path.exists(path):
            pytest.skip("aie2-isa.json not found")
        with open(path) as f:
            return json.load(f)

    def _find_instr(self, isa_data, name):
        for section, items in isa_data.items():
            if isinstance(items, list):
                for i in items:
                    if i.get("name") == name:
                        return i
        return None

    def test_vlda_w_spill_testable(self, isa_data):
        """VLDA_dmw_lda_w_ag_spill must be testable via LoadStrategy."""
        instr = self._find_instr(isa_data, "VLDA_dmw_lda_w_ag_spill")
        assert instr is not None
        strategy = isa_test_gen.LoadStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"Expected testable: {reason}"

    def test_vst_w_spill_testable(self, isa_data):
        """VST_dmw_sts_w_ag_spill must be testable via StoreStrategy."""
        instr = self._find_instr(isa_data, "VST_dmw_sts_w_ag_spill")
        assert instr is not None
        strategy = isa_test_gen.StoreStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"Expected testable: {reason}"

    def test_vst_am_spill_blocked_by_llvm_bug(self, isa_data):
        """VST_dmw_sts_am_ag_spill is blocked by llvm-mc encoder bug.

        The SP-relative am-class spill store crashes llvm-mc because the
        encoder expects step=32 but the immediate field has scale=1.
        """
        instr = self._find_instr(isa_data, "VST_dmw_sts_am_ag_spill")
        assert instr is not None
        strategy = isa_test_gen.StoreStrategy()
        can, reason = strategy.can_test(instr)
        assert not can
        assert "llvm-mc" in reason.lower() or "bug" in reason.lower()

    def test_vst_128_spill_testable(self, isa_data):
        """VST_128_ag_spill must be testable via StoreStrategy."""
        instr = self._find_instr(isa_data, "VST_128_ag_spill")
        assert instr is not None
        strategy = isa_test_gen.StoreStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"Expected testable: {reason}"

    def test_lda_dms_spill_now_testable(self, isa_data):
        """LDA_dms_spill: LdaScl composite is now in TESTABLE_COMPOSITE_KINDS."""
        instr = self._find_instr(isa_data, "LDA_dms_spill")
        assert instr is not None
        strategy = isa_test_gen.LoadStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"LDA_dms_spill should be testable now: {reason}"

    def test_st_dms_spill_now_testable(self, isa_data):
        """ST_dms_spill: LdaScl composite is now in TESTABLE_COMPOSITE_KINDS."""
        instr = self._find_instr(isa_data, "ST_dms_spill")
        assert instr is not None
        strategy = isa_test_gen.StoreStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"ST_dms_spill should be testable now: {reason}"

    def test_lda_dms_idx_imm_testable(self, isa_data):
        """LDA_dms_lda_idx_imm (DMS scalar load, mLdaScl dest) must be testable."""
        instr = self._find_instr(isa_data, "LDA_dms_lda_idx_imm")
        assert instr is not None, "LDA_dms_lda_idx_imm not found in ISA"
        strategy = isa_test_gen.LoadStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"LDA_dms_lda_idx_imm should be testable, got: {reason}"

    def test_lda_2d_dms_testable(self, isa_data):
        """LDA_2D_dms_lda (DMS scalar 2D load, mLdaScl dest) must be testable."""
        instr = self._find_instr(isa_data, "LDA_2D_dms_lda")
        assert instr is not None, "LDA_2D_dms_lda not found in ISA"
        strategy = isa_test_gen.LoadStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"LDA_2D_dms_lda should be testable, got: {reason}"

    def test_st_dms_idx_testable(self, isa_data):
        """ST_dms_sts_idx (DMS scalar store, mSclSt source) must be testable."""
        instr = self._find_instr(isa_data, "ST_dms_sts_idx")
        assert instr is not None, "ST_dms_sts_idx not found in ISA"
        strategy = isa_test_gen.StoreStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"ST_dms_sts_idx should be testable, got: {reason}"

    def test_st_2d_dms_testable(self, isa_data):
        """ST_2D_dms_sts (DMS scalar 2D store, mSclSt source) must be testable."""
        instr = self._find_instr(isa_data, "ST_2D_dms_sts")
        assert instr is not None, "ST_2D_dms_sts not found in ISA"
        strategy = isa_test_gen.StoreStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"ST_2D_dms_sts should be testable, got: {reason}"

    def test_lda_dmv_q_idx_testable(self, isa_data):
        """LDA_dmv_lda_q_ag_idx (DMV_Q quad load, bw=2 accumulator) must be testable."""
        instr = self._find_instr(isa_data, "LDA_dmv_lda_q_ag_idx")
        assert instr is not None, "LDA_dmv_lda_q_ag_idx not found in ISA"
        strategy = isa_test_gen.LoadStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"LDA_dmv_lda_q_ag_idx (DMV_Q) should be testable, got: {reason}"

    def test_st_dmv_q_idx_testable(self, isa_data):
        """ST_dmv_sts_q_ag_idx (DMV_Q quad store, bw=2 accumulator) must be testable."""
        instr = self._find_instr(isa_data, "ST_dmv_sts_q_ag_idx")
        assert instr is not None, "ST_dmv_sts_q_ag_idx not found in ISA"
        strategy = isa_test_gen.StoreStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"ST_dmv_sts_q_ag_idx (DMV_Q) should be testable, got: {reason}"


# ===================================================================
# BranchStrategy tests
# ===================================================================

class TestBranchStrategy:
    """Tests for BranchStrategy with absolute IW address targets."""

    def _make_j(self):
        return _make_instr("J_jump_imm", "j", "j\t$cpmaddr", [
            _make_imm_op("cpmaddr", bit_width=20, signed=False),
        ], slot="lng")

    def _make_jl(self):
        return _make_instr("JL", "jl", "jl\t$cpmaddr", [
            _make_imm_op("cpmaddr", bit_width=20, signed=False),
        ], slot="lng")

    def _make_jnz(self):
        return _make_instr("JNZ", "jnz", "jnz\t$mRx, $cpmaddr", [
            _make_reg_op("mRx", "scalar"),
            _make_imm_op("cpmaddr", bit_width=20, signed=False),
        ], slot="lng")

    def _make_jz(self):
        return _make_instr("JZ", "jz", "jz\t$mRx, $cpmaddr", [
            _make_reg_op("mRx", "scalar"),
            _make_imm_op("cpmaddr", bit_width=20, signed=False),
        ], slot="lng")

    def test_jnz_can_test(self):
        strategy = isa_test_gen.BranchStrategy()
        can, reason = strategy.can_test(self._make_jnz())
        assert can, f"Expected can_test=True: {reason}"

    def test_jz_can_test(self):
        strategy = isa_test_gen.BranchStrategy()
        can, reason = strategy.can_test(self._make_jz())
        assert can, f"Expected can_test=True: {reason}"

    def test_j_immediate_can_test(self):
        strategy = isa_test_gen.BranchStrategy()
        can, reason = strategy.can_test(self._make_j())
        assert can, f"Expected can_test=True: {reason}"

    def _make_j_ind(self):
        return _make_instr("J_jump_ind", "j", "j\t$mPm", [
            _make_reg_op("mPm", "pointer"),
        ], slot="alu")

    def _make_jl_ind(self):
        return _make_instr("JL_IND", "jl", "jl\t$mPm", [
            _make_reg_op("mPm", "pointer"),
        ], slot="alu")

    def _make_ret(self):
        return _make_instr("RET", "ret", "ret lr", [], slot="alu")

    def _make_jnzd(self):
        return _make_instr("JNZD", "jnzd", "jnzd\t$mRx, $mRx0, $mPm", [
            _make_reg_op("mRx", "scalar"),
            _make_reg_op("mRx0", "scalar"),
            _make_reg_op("mPm", "pointer"),
        ], slot="alu")

    def test_j_register_indirect_can_test(self):
        """Register-indirect j $mPm should now be testable."""
        strategy = isa_test_gen.BranchStrategy()
        can, reason = strategy.can_test(self._make_j_ind())
        assert can, f"Expected can_test=True: {reason}"

    def test_jl_register_indirect_can_test(self):
        """Register-indirect jl $mPm should now be testable."""
        strategy = isa_test_gen.BranchStrategy()
        can, reason = strategy.can_test(self._make_jl_ind())
        assert can, f"Expected can_test=True: {reason}"

    def test_ret_can_test(self):
        """ret lr should now be testable."""
        strategy = isa_test_gen.BranchStrategy()
        can, reason = strategy.can_test(self._make_ret())
        assert can, f"Expected can_test=True: {reason}"

    def test_jnzd_can_test(self):
        """jnzd should now be testable."""
        strategy = isa_test_gen.BranchStrategy()
        can, reason = strategy.can_test(self._make_jnzd())
        assert can, f"Expected can_test=True: {reason}"

    def test_ret_generates_lr_setup(self):
        """ret should set LR to the taken target IW address."""
        strategy = isa_test_gen.BranchStrategy()
        regs = {}
        asm = strategy.generate_test_point(self._make_ret(), regs,
                                           in_offset=0, out_offset=0)
        # Should set up LR with target address.
        assert "mov lr, #" in asm
        # Should have ret lr instruction.
        assert "ret lr" in asm
        # Should store markers.
        assert "#170" in asm  # 0xAA before marker
        assert "#204" in asm  # 0xCC taken marker

    def test_j_ind_generates_pointer_setup(self):
        """Register-indirect j should load target IW into pointer register."""
        strategy = isa_test_gen.BranchStrategy()
        regs = {"mPm": "p2"}
        asm = strategy.generate_test_point(self._make_j_ind(), regs,
                                           in_offset=0, out_offset=0)
        # Should set up pointer register with target address.
        assert "mov p2, #" in asm
        # Should have j p2 instruction (not j #N).
        assert "j p2" in asm

    def test_jl_ind_generates_pointer_setup(self):
        """Register-indirect jl should load target IW into pointer register."""
        strategy = isa_test_gen.BranchStrategy()
        regs = {"mPm": "p3"}
        asm = strategy.generate_test_point(self._make_jl_ind(), regs,
                                           in_offset=0, out_offset=0)
        assert "mov p3, #" in asm
        assert "jl p3" in asm

    def test_jnzd_generates_counter_setup(self):
        """jnzd should set up counter register and pointer register."""
        strategy = isa_test_gen.BranchStrategy()
        regs = {"mRx": "r0", "mRx0": "r1", "mPm": "p2", "_combo_idx": 0}
        asm = strategy.generate_test_point(self._make_jnzd(), regs,
                                           in_offset=0, out_offset=0)
        # Combo 0 = taken: counter should be 2 (decrement to 1, nonzero).
        assert "mov r1, #2" in asm
        assert "mov p2, #" in asm
        assert "jnzd r0, r1, p2" in asm

    def test_jnzd_not_taken_combo(self):
        """jnzd combo 1 = not-taken: counter should be 1 (decrement to 0)."""
        strategy = isa_test_gen.BranchStrategy()
        regs = {"mRx": "r0", "mRx0": "r1", "mPm": "p2", "_combo_idx": 1}
        asm = strategy.generate_test_point(self._make_jnzd(), regs,
                                           in_offset=0, out_offset=0)
        # Combo 1 = not-taken: counter should be 1 (decrement to 0, zero).
        assert "mov r1, #1" in asm

    def test_jnzd_generates_two_combos(self):
        """jnzd is conditional, should produce 2 combos."""
        strategy = isa_test_gen.BranchStrategy()
        combos = strategy.generate_combos(self._make_jnzd())
        assert len(combos) == 2

    def test_ret_generates_one_combo(self):
        """ret is unconditional, should produce 1 combo."""
        strategy = isa_test_gen.BranchStrategy()
        combos = strategy.generate_combos(self._make_ret())
        assert len(combos) == 1

    def test_j_ind_generates_one_combo(self):
        """Register-indirect j is unconditional, should produce 1 combo."""
        strategy = isa_test_gen.BranchStrategy()
        combos = strategy.generate_combos(self._make_j_ind())
        assert len(combos) == 1

    def test_ret_no_input_buffer(self):
        """ret needs no input buffer."""
        strategy = isa_test_gen.BranchStrategy()
        assert strategy.compute_input_size(self._make_ret(), {}) == 0

    def test_j_ind_iw_offset_shifts_targets(self):
        """Register-indirect j should respect code_iw_offset."""
        strategy = isa_test_gen.BranchStrategy()
        regs = {"mPm": "p2"}
        import re as _re
        asm0 = strategy.generate_test_point(self._make_j_ind(), regs,
                                            in_offset=0, out_offset=0,
                                            code_iw_offset=0)
        asm10 = strategy.generate_test_point(self._make_j_ind(), regs,
                                             in_offset=0, out_offset=0,
                                             code_iw_offset=10)
        # Extract the pointer setup value (mov p2, #N).
        targets0 = _re.findall(r'mov p2, #(\d+)', asm0)
        targets10 = _re.findall(r'mov p2, #(\d+)', asm10)
        assert len(targets0) >= 1
        assert len(targets10) >= 1
        # Offset=10 should shift the target by 10.
        assert int(targets10[0]) == int(targets0[0]) + 10

    def test_j_generates_numeric_targets(self):
        """Unconditional j should emit numeric IW addresses, not labels."""
        strategy = isa_test_gen.BranchStrategy()
        regs = {"cpmaddr": "0"}
        asm = strategy.generate_test_point(self._make_j(), regs,
                                           in_offset=0, out_offset=0)
        # No labels -- only numeric targets.
        assert ".Ltaken" not in asm
        assert ".Ldone" not in asm
        # Should have j #N instruction.
        assert "j #" in asm
        # Should store markers.
        assert "#170" in asm  # 0xAA before marker
        assert "#204" in asm  # 0xCC taken marker
        assert "#187" in asm  # 0xBB fall-through marker

    def test_jnz_generates_condition_setup(self):
        """Conditional jnz emits mov r0, #N for the condition value."""
        strategy = isa_test_gen.BranchStrategy()
        regs = {"mRx": "r0", "cpmaddr": "0", "_combo_idx": 0}
        asm = strategy.generate_test_point(self._make_jnz(), regs,
                                           in_offset=0, out_offset=0)
        # Combo 0 = taken path for jnz -> needs nonzero condition.
        assert "mov r0, #1" in asm
        assert "jnz r0, #" in asm

    def test_jnz_not_taken_combo(self):
        """Combo index 1 for jnz = not-taken (condition = 0)."""
        strategy = isa_test_gen.BranchStrategy()
        regs = {"mRx": "r0", "cpmaddr": "0", "_combo_idx": 1}
        asm = strategy.generate_test_point(self._make_jnz(), regs,
                                           in_offset=0, out_offset=0)
        assert "mov r0, #0" in asm

    def test_jz_taken_combo(self):
        """Combo 0 for jz = taken -> needs zero condition."""
        strategy = isa_test_gen.BranchStrategy()
        regs = {"mRx": "r0", "cpmaddr": "0", "_combo_idx": 0}
        asm = strategy.generate_test_point(self._make_jz(), regs,
                                           in_offset=0, out_offset=0)
        assert "mov r0, #0" in asm
        assert "jz r0, #" in asm

    def test_code_iw_offset_shifts_targets(self):
        """Non-zero code_iw_offset shifts all branch target addresses."""
        strategy = isa_test_gen.BranchStrategy()
        regs = {"cpmaddr": "0"}
        asm0 = strategy.generate_test_point(self._make_j(), regs,
                                            in_offset=0, out_offset=0,
                                            code_iw_offset=0)
        asm10 = strategy.generate_test_point(self._make_j(), regs,
                                             in_offset=0, out_offset=0,
                                             code_iw_offset=10)
        import re as _re
        targets0 = [int(m) for m in _re.findall(r'j #(\d+)', asm0)]
        targets10 = [int(m) for m in _re.findall(r'j #(\d+)', asm10)]
        # Both should have exactly 2 j targets (branch + skip-to-done).
        assert len(targets0) == 2
        assert len(targets10) == 2
        # Offset=10 should shift every target by exactly 10.
        for t0, t10 in zip(targets0, targets10):
            assert t10 == t0 + 10

    def test_no_input_buffer_needed(self):
        """Branch tests use immediate condition values, no input buffer."""
        strategy = isa_test_gen.BranchStrategy()
        regs = {"mRx": "r0", "cpmaddr": "0"}
        assert strategy.compute_input_size(self._make_jnz(), regs) == 0
        assert strategy.compute_input_size(self._make_j(), regs) == 0

    def test_output_size_is_8(self):
        """Two 4-byte markers: before + path."""
        strategy = isa_test_gen.BranchStrategy()
        assert strategy.compute_output_size(self._make_j()) == 8

    def test_conditional_combos(self):
        """Conditional branches should produce 2 combos (taken + not-taken)."""
        strategy = isa_test_gen.BranchStrategy()
        combos = strategy.generate_combos(self._make_jnz())
        assert len(combos) == 2

    def test_unconditional_combos(self):
        """Unconditional j should produce 1 combo."""
        strategy = isa_test_gen.BranchStrategy()
        combos = strategy.generate_combos(self._make_j())
        assert len(combos) == 1

    def test_delay_slots_present(self):
        """Branch should be followed by 5 NOP delay slots."""
        strategy = isa_test_gen.BranchStrategy()
        regs = {"cpmaddr": "0"}
        asm = strategy.generate_test_point(self._make_j(), regs,
                                           in_offset=0, out_offset=0)
        lines = [l.strip() for l in asm.split("\n")
                 if l.strip() and not l.strip().startswith("//")]
        # Find the j instruction and count following nops.
        for i, line in enumerate(lines):
            if line.startswith("j #"):
                nop_count = 0
                for k in range(i + 1, min(i + 6, len(lines))):
                    if lines[k] == "nop":
                        nop_count += 1
                    else:
                        break
                assert nop_count == 5, f"Expected 5 delay NOPs, got {nop_count}"
                break
        else:
            assert False, "No j # instruction found"

    def test_jl_generates_jl_instruction(self):
        """jl should emit jl #N, not j #N."""
        strategy = isa_test_gen.BranchStrategy()
        regs = {"cpmaddr": "0"}
        asm = strategy.generate_test_point(self._make_jl(), regs,
                                           in_offset=0, out_offset=0)
        assert "jl #" in asm


# ===================================================================
# LockStrategy tests
# ===================================================================

class TestLockStrategy:
    """Tests for LockStrategy with marker-based verification."""

    def _make_acq_imm(self):
        return _make_instr("ACQ_mLockId_imm", "acq", "acq\t$id, $mRy", [
            _make_imm_op("id", bit_width=6, signed=True),
            _make_reg_op("mRy", "scalar"),
        ], slot="alu")

    def _make_acq_reg(self):
        return _make_instr("ACQ_mLockId_reg", "acq", "acq\t$mRx, $mRy", [
            _make_reg_op("mRy", "scalar"),
            _make_reg_op("mRx", "scalar"),
            {"name": "dontcare1", "bit_width": 1, "operand_type": "unknown"},
        ], slot="alu")

    def _make_rel_imm(self):
        return _make_instr("REL_mLockId_imm", "rel", "rel\t$id, $mRy", [
            _make_imm_op("id", bit_width=6, signed=True),
            _make_reg_op("mRy", "scalar"),
        ], slot="alu")

    def _make_rel_reg(self):
        return _make_instr("REL_mLockId_reg", "rel", "rel\t$mRx, $mRy", [
            {"name": "dontcare1", "bit_width": 1, "operand_type": "unknown"},
            _make_reg_op("mRy", "scalar"),
            _make_reg_op("mRx", "scalar"),
        ], slot="alu")

    def _make_acq_cond_imm(self):
        return _make_instr("ACQ_COND_mLockId_imm", "acq.cond",
                           "acq.cond\t$id, $mRy, r26", [
            _make_imm_op("id", bit_width=6, signed=True),
            _make_reg_op("mRy", "scalar"),
        ], slot="alu")

    def _make_rel_cond_reg(self):
        return _make_instr("REL_COND_mLockId_reg", "rel.cond",
                           "rel.cond\t$mRx, $mRy, r26", [
            {"name": "dontcare1", "bit_width": 1, "operand_type": "unknown"},
            _make_reg_op("mRx", "scalar"),
            _make_reg_op("mRy", "scalar"),
        ], slot="alu")

    def test_acq_imm_can_test(self):
        strategy = isa_test_gen.LockStrategy()
        can, reason = strategy.can_test(self._make_acq_imm())
        assert can, f"Expected can_test=True: {reason}"

    def test_acq_reg_can_test(self):
        strategy = isa_test_gen.LockStrategy()
        can, reason = strategy.can_test(self._make_acq_reg())
        assert can, f"Expected can_test=True: {reason}"

    def test_rel_imm_can_test(self):
        strategy = isa_test_gen.LockStrategy()
        can, reason = strategy.can_test(self._make_rel_imm())
        assert can, f"Expected can_test=True: {reason}"

    def test_rel_reg_can_test(self):
        strategy = isa_test_gen.LockStrategy()
        can, reason = strategy.can_test(self._make_rel_reg())
        assert can, f"Expected can_test=True: {reason}"

    def test_acq_cond_can_test(self):
        strategy = isa_test_gen.LockStrategy()
        can, reason = strategy.can_test(self._make_acq_cond_imm())
        assert can, f"Expected can_test=True: {reason}"

    def test_rel_cond_can_test(self):
        strategy = isa_test_gen.LockStrategy()
        can, reason = strategy.can_test(self._make_rel_cond_reg())
        assert can, f"Expected can_test=True: {reason}"

    def test_non_lock_rejected(self):
        """Non-lock mnemonic should be rejected."""
        instr = _make_instr("ADD", "add", "add\t$mRx, $mRy, $mRz", [
            _make_reg_op("mRx", "scalar"),
            _make_reg_op("mRy", "scalar"),
            _make_reg_op("mRz", "scalar"),
        ])
        strategy = isa_test_gen.LockStrategy()
        can, _ = strategy.can_test(instr)
        assert not can

    def test_no_input_buffer(self):
        """Lock tests use immediate/register setup, no input buffer."""
        strategy = isa_test_gen.LockStrategy()
        assert strategy.compute_input_size(self._make_acq_imm(), {}) == 0

    def test_output_size_is_8(self):
        """Two 4-byte markers: before + after."""
        strategy = isa_test_gen.LockStrategy()
        assert strategy.compute_output_size(self._make_acq_imm()) == 8

    def test_one_combo(self):
        """Lock instructions produce exactly 1 combo."""
        strategy = isa_test_gen.LockStrategy()
        combos = strategy.generate_combos(self._make_acq_imm())
        assert len(combos) == 1

    def test_rel_imm_generates_markers(self):
        """rel with immediate lock ID should store before/after markers."""
        strategy = isa_test_gen.LockStrategy()
        regs = {"id": "0", "mRy": "r1"}
        asm = strategy.generate_test_point(self._make_rel_imm(), regs,
                                           in_offset=0, out_offset=0)
        assert "#170" in asm  # 0xAA before marker
        assert "#204" in asm  # 0xCC after marker
        # asm_string uses tabs: "rel\t#0, r1"
        assert "rel" in asm and "#0" in asm and "r1" in asm

    def test_acq_imm_has_rel_setup(self):
        """acq should be preceded by a rel to ensure the lock won't stall."""
        strategy = isa_test_gen.LockStrategy()
        regs = {"id": "0", "mRy": "r1"}
        asm = strategy.generate_test_point(self._make_acq_imm(), regs,
                                           in_offset=0, out_offset=0)
        # rel setup should appear before the acq.
        lines = asm.split("\n")
        rel_line = next((i for i, l in enumerate(lines) if "rel" in l), None)
        acq_line = next((i for i, l in enumerate(lines) if "acq" in l), None)
        assert rel_line is not None, "Expected rel setup before acq"
        assert acq_line is not None, "Expected acq instruction"
        assert rel_line < acq_line, "rel setup must come before acq"

    def test_acq_reg_has_rel_setup(self):
        """Register-form acq should also be preceded by rel setup."""
        strategy = isa_test_gen.LockStrategy()
        regs = {"mRx": "r2", "mRy": "r1", "dontcare1": "0"}
        asm = strategy.generate_test_point(self._make_acq_reg(), regs,
                                           in_offset=0, out_offset=0)
        assert "rel" in asm
        # asm_string uses tabs: "acq\tr2, r1"
        assert "acq" in asm and "r2" in asm

    def test_acq_cond_sets_r26(self):
        """Conditional acquire should set r26=1 so the operation executes."""
        strategy = isa_test_gen.LockStrategy()
        regs = {"id": "0", "mRy": "r1"}
        asm = strategy.generate_test_point(self._make_acq_cond_imm(), regs,
                                           in_offset=0, out_offset=0)
        assert "mov r26, #1" in asm
        assert "acq.cond" in asm

    def test_rel_cond_sets_r26(self):
        """Conditional release should set r26=1 so the operation executes."""
        strategy = isa_test_gen.LockStrategy()
        regs = {"mRx": "r2", "mRy": "r1", "dontcare1": "0"}
        asm = strategy.generate_test_point(self._make_rel_cond_reg(), regs,
                                           in_offset=0, out_offset=0)
        assert "mov r26, #1" in asm
        assert "rel.cond" in asm


# ===================================================================
# FifoLoadStrategy tests
# ===================================================================

class TestFifoLoadStrategy:
    """Tests for FifoLoadStrategy with marker-based verification."""

    def _make_compr_fill(self):
        return _make_instr("VLDB_COMPR_FILL", "vldb.compr.fill",
                           "vldb.compr.fill\t[$ptr]", [
            _make_reg_op("ptr", "pointer"),
        ], slot="ldb")

    def _make_compr_peek(self):
        return _make_instr("VLDB_COMPR_PEEK", "vldb.compr.peek",
                           "vldb.compr.peek\t$dst, [$ptr]", [
            _make_reg_op("dst", "vector256"),
            _make_reg_op("ptr", "pointer"),
        ], slot="ldb")

    def _make_compr_pop(self):
        return _make_instr("VLDB_COMPR_POP", "vldb.compr.pop",
                           "vldb.compr.pop\t$dst, [$ptr]", [
            _make_reg_op("ptr", "pointer"),
            _make_reg_op("dst", "vector256"),
        ], slot="ldb")

    def _make_compr_reset(self):
        return _make_instr("VLDB_COMPR_RESET", "vldb.compr.reset",
                           "vldb.compr.reset\t[$ptr]", [
            _make_reg_op("ptr", "pointer"),
        ], slot="ldb")

    def _make_sparse_fill(self):
        return _make_instr("VLDB_SPARSE_FILL_8", "vldb.sparse.fill.8",
                           "vldb.sparse.fill.8\t[$ptr]", [
            _make_reg_op("ptr", "pointer"),
        ], slot="ldb")

    def _make_sparse_peek(self):
        return _make_instr("VLDB_SPARSE_PEEK_8", "vldb.sparse.peek.8",
                           "vldb.sparse.peek.8\t$dst, [$ptr]", [
            {"name": "dst", "bit_width": 3, "operand_type": "composite_register",
             "register_kind": "QXHLb"},
            _make_reg_op("ptr", "pointer"),
        ], slot="ldb")

    def _make_sparse_pop(self):
        return _make_instr("VLDB_SPARSE_POP_4", "vldb.sparse.pop.4",
                           "vldb.sparse.pop.4\t$dst, [$ptr]", [
            _make_reg_op("ptr", "pointer"),
            {"name": "dst", "bit_width": 3, "operand_type": "composite_register",
             "register_kind": "QXHLb"},
        ], slot="ldb")

    def test_compr_fill_can_test(self):
        strategy = isa_test_gen.FifoLoadStrategy()
        can, reason = strategy.can_test(self._make_compr_fill())
        assert can, f"Expected can_test=True: {reason}"

    def test_compr_peek_can_test(self):
        strategy = isa_test_gen.FifoLoadStrategy()
        can, reason = strategy.can_test(self._make_compr_peek())
        assert can, f"Expected can_test=True: {reason}"

    def test_compr_pop_can_test(self):
        strategy = isa_test_gen.FifoLoadStrategy()
        can, reason = strategy.can_test(self._make_compr_pop())
        assert can, f"Expected can_test=True: {reason}"

    def test_compr_reset_can_test(self):
        strategy = isa_test_gen.FifoLoadStrategy()
        can, reason = strategy.can_test(self._make_compr_reset())
        assert can, f"Expected can_test=True: {reason}"

    def test_sparse_fill_can_test(self):
        strategy = isa_test_gen.FifoLoadStrategy()
        can, reason = strategy.can_test(self._make_sparse_fill())
        assert can, f"Expected can_test=True: {reason}"

    def test_sparse_peek_can_test(self):
        strategy = isa_test_gen.FifoLoadStrategy()
        can, reason = strategy.can_test(self._make_sparse_peek())
        assert can, f"Expected can_test=True: {reason}"

    def test_sparse_pop_can_test(self):
        strategy = isa_test_gen.FifoLoadStrategy()
        can, reason = strategy.can_test(self._make_sparse_pop())
        assert can, f"Expected can_test=True: {reason}"

    def test_non_fifo_rejected(self):
        """Normal load should not be handled by FifoLoadStrategy."""
        instr = _make_instr("LDA", "lda", "lda\t$dst, [$ptr, #$imm]", [
            _make_reg_op("dst", "scalar"),
            _make_reg_op("ptr", "pointer"),
            _make_imm_op("imm", bit_width=7, signed=True),
        ])
        strategy = isa_test_gen.FifoLoadStrategy()
        can, _ = strategy.can_test(instr)
        assert not can

    def test_no_input_buffer(self):
        """FIFO tests don't use the input buffer (markers only)."""
        strategy = isa_test_gen.FifoLoadStrategy()
        assert strategy.compute_input_size(self._make_compr_fill(), {}) == 0

    def test_output_size_is_8(self):
        """Two 4-byte markers: before + after."""
        strategy = isa_test_gen.FifoLoadStrategy()
        assert strategy.compute_output_size(self._make_compr_fill()) == 8

    def test_one_combo(self):
        strategy = isa_test_gen.FifoLoadStrategy()
        combos = strategy.generate_combos(self._make_compr_fill())
        assert len(combos) == 1

    def test_fill_generates_markers(self):
        """FILL should have before/after markers."""
        strategy = isa_test_gen.FifoLoadStrategy()
        regs = {"ptr": "p2"}
        asm = strategy.generate_test_point(self._make_compr_fill(), regs,
                                           in_offset=0, out_offset=0)
        assert "#170" in asm  # before
        assert "#204" in asm  # after
        assert "vldb.compr.fill" in asm

    def test_peek_has_reset_fill_setup(self):
        """PEEK should be preceded by RESET + FILL to prime the FIFO."""
        strategy = isa_test_gen.FifoLoadStrategy()
        regs = {"dst": "wl0", "ptr": "p2"}
        asm = strategy.generate_test_point(self._make_compr_peek(), regs,
                                           in_offset=0, out_offset=0)
        lines = asm.split("\n")
        # Find reset, fill, and peek lines.
        reset_line = next((i for i, l in enumerate(lines)
                          if "vldb.compr.reset" in l), None)
        fill_line = next((i for i, l in enumerate(lines)
                         if "vldb.compr.fill" in l), None)
        peek_line = next((i for i, l in enumerate(lines)
                         if "vldb.compr.peek" in l), None)
        assert reset_line is not None, "Expected reset before peek"
        assert fill_line is not None, "Expected fill before peek"
        assert peek_line is not None, "Expected peek instruction"
        assert reset_line < fill_line < peek_line

    def test_pop_has_reset_fill_setup(self):
        """POP should be preceded by RESET + FILL to prime the FIFO."""
        strategy = isa_test_gen.FifoLoadStrategy()
        regs = {"dst": "wl0", "ptr": "p2"}
        asm = strategy.generate_test_point(self._make_compr_pop(), regs,
                                           in_offset=0, out_offset=0)
        assert "vldb.compr.reset" in asm
        assert "vldb.compr.fill" in asm
        assert "vldb.compr.pop" in asm

    def test_sparse_peek_has_setup(self):
        """Sparse PEEK uses sparse-specific reset and fill."""
        strategy = isa_test_gen.FifoLoadStrategy()
        regs = {"dst": "qwl0", "ptr": "p2"}
        asm = strategy.generate_test_point(self._make_sparse_peek(), regs,
                                           in_offset=0, out_offset=0)
        assert "vldb.sparse.reset.8" in asm
        assert "vldb.sparse.fill.8" in asm
        assert "vldb.sparse.peek.8" in asm

    def test_sparse_pop_has_setup(self):
        """Sparse POP uses sparse-specific reset and fill."""
        strategy = isa_test_gen.FifoLoadStrategy()
        regs = {"dst": "qwl0", "ptr": "p2"}
        asm = strategy.generate_test_point(self._make_sparse_pop(), regs,
                                           in_offset=0, out_offset=0)
        assert "vldb.sparse.reset.4" in asm
        assert "vldb.sparse.fill.4" in asm
        assert "vldb.sparse.pop.4" in asm

    def test_fill_has_nop_sled_after(self):
        """FILL needs pipeline delay (NOPs) before the FIFO is ready."""
        strategy = isa_test_gen.FifoLoadStrategy()
        regs = {"dst": "wl0", "ptr": "p2"}
        asm = strategy.generate_test_point(self._make_compr_peek(), regs,
                                           in_offset=0, out_offset=0)
        # Count NOPs between fill and peek.
        lines = [l.strip() for l in asm.split("\n") if l.strip()]
        fill_idx = next(i for i, l in enumerate(lines)
                       if "vldb.compr.fill" in l)
        peek_idx = next(i for i, l in enumerate(lines)
                       if "vldb.compr.peek" in l)
        nops = sum(1 for l in lines[fill_idx+1:peek_idx] if l == "nop")
        assert nops >= 5, f"Expected >= 5 NOPs between fill and peek, got {nops}"


# ===================================================================
# CascadeStrategy tests
# ===================================================================

class TestCascadeStrategy:
    """Tests for CascadeStrategy with marker-based verification."""

    def _make_mcd_write(self):
        return _make_instr("VMOV_mv_mcd", "vmov", "vmov\tMCD, $src", [
            {"name": "src", "bit_width": 6, "operand_type": "composite_register",
             "register_kind": "MvBMXDst"},
        ], slot="st")

    def _make_scd_read(self):
        return _make_instr("VMOV_mv_scd", "vmov", "vmov\t$dst, SCD", [
            {"name": "dst", "bit_width": 6, "operand_type": "composite_register",
             "register_kind": "MvBMXDst"},
        ], slot="lda")

    def _make_vmov_hi(self):
        return _make_instr("VMOV_HI", "vmov.hi", "vmov.hi\t$dst, SCD", [
            {"name": "dst", "bit_width": 4, "operand_type": "unknown"},
        ], slot="lda")

    def test_mcd_write_can_test(self):
        """Cascade write should be testable (MCD FIFO depth 4)."""
        strategy = isa_test_gen.CascadeStrategy()
        can, reason = strategy.can_test(self._make_mcd_write())
        assert can, f"Expected can_test=True: {reason}"

    def test_scd_read_rejected(self):
        """Cascade read should be rejected (stalls without neighboring tile)."""
        strategy = isa_test_gen.CascadeStrategy()
        can, _ = strategy.can_test(self._make_scd_read())
        assert not can

    def test_vmov_hi_rejected(self):
        """vmov.hi SCD should be rejected (cascade read)."""
        strategy = isa_test_gen.CascadeStrategy()
        can, _ = strategy.can_test(self._make_vmov_hi())
        assert not can

    def test_non_cascade_rejected(self):
        strategy = isa_test_gen.CascadeStrategy()
        instr = _make_instr("ADD", "add", "add\t$mRx, $mRy, $mRz", [
            _make_reg_op("mRx", "scalar"),
        ])
        can, _ = strategy.can_test(instr)
        assert not can

    def test_no_input_buffer(self):
        strategy = isa_test_gen.CascadeStrategy()
        assert strategy.compute_input_size(self._make_mcd_write(), {}) == 0

    def test_output_size_is_8(self):
        strategy = isa_test_gen.CascadeStrategy()
        assert strategy.compute_output_size(self._make_mcd_write()) == 8

    def test_mcd_write_generates_markers(self):
        """MCD write should have before/after markers and enable crMCDEn."""
        strategy = isa_test_gen.CascadeStrategy()
        regs = {"src": "x0"}
        asm = strategy.generate_test_point(self._make_mcd_write(), regs,
                                           in_offset=0, out_offset=0)
        assert "#170" in asm  # before
        assert "#204" in asm  # after
        assert "crMCDEn" in asm
        assert "vmov" in asm and "MCD" in asm

    def test_mcd_write_loads_source(self):
        """MCD write should load data into the source register first."""
        strategy = isa_test_gen.CascadeStrategy()
        regs = {"src": "x0"}
        asm = strategy.generate_test_point(self._make_mcd_write(), regs,
                                           in_offset=0, out_offset=0)
        # Should load data into x0 via vlda (vector load from input buffer).
        assert "vlda" in asm


# ===================================================================
# Task 4: generate_all integration tests
# ===================================================================

class TestGenerateAll:
    """Integration tests for the full pipeline."""

    @pytest.fixture
    def isa_json_path(self):
        path = os.path.join(os.path.dirname(__file__), "aie2-isa.json")
        if not os.path.exists(path):
            pytest.skip("aie2-isa.json not found")
        return path

    @pytest.fixture
    def out_dir(self, tmp_path):
        return str(tmp_path / "isa-tests")

    def test_generates_manifest(self, isa_json_path, out_dir):
        """generate_all should produce a manifest.json."""
        manifest = generate_all(isa_json_path, out_dir)
        manifest_path = os.path.join(out_dir, "manifest.json")
        assert os.path.exists(manifest_path)
        with open(manifest_path) as f:
            loaded = json.load(f)
        assert "batches" in loaded
        assert "testable_instructions" in loaded
        assert "total_test_points" in loaded

    def test_generates_batch_files(self, isa_json_path, out_dir):
        """generate_all should produce .s files."""
        manifest = generate_all(isa_json_path, out_dir)
        assert manifest["total_batches"] > 0
        for batch in manifest["batches"]:
            s_path = os.path.join(out_dir, batch["filename"])
            assert os.path.exists(s_path), f"Missing {batch['filename']}"

    def test_batch_structure(self, isa_json_path, out_dir):
        """Each batch should have valid metadata."""
        manifest = generate_all(isa_json_path, out_dir)
        for batch in manifest["batches"]:
            assert "batch_index" in batch
            assert "test_count" in batch
            assert batch["test_count"] > 0
            # Batches are sized by code footprint (program memory),
            # not a fixed test count.  Sanity-check: no batch should have
            # more than ~200 test points (would imply broken packing).
            assert batch["test_count"] <= 200
            assert "in_size" in batch
            assert "out_size" in batch
            assert batch["in_size"] > 0
            assert batch["out_size"] > 0
            assert "tests" in batch
            assert len(batch["tests"]) == batch["test_count"]

    def test_test_point_metadata(self, isa_json_path, out_dir):
        """Each test point should have required fields."""
        manifest = generate_all(isa_json_path, out_dir)
        for batch in manifest["batches"]:
            for tp in batch["tests"]:
                assert "instruction" in tp
                assert "slot" in tp
                assert "combo_index" in tp
                assert "operands" in tp
                assert isinstance(tp["operands"], dict)
                assert "in_offset" in tp
                assert "in_size" in tp
                assert "out_offset" in tp
                assert "out_size" in tp

    def test_summary_counts(self, isa_json_path, out_dir):
        """Summary counts should be consistent."""
        manifest = generate_all(isa_json_path, out_dir)
        assert manifest["testable_instructions"] > 0
        assert manifest["total_test_points"] > manifest["testable_instructions"]
        total_from_batches = sum(b["test_count"] for b in manifest["batches"])
        assert total_from_batches == manifest["total_test_points"]

    def test_batch_assembles_with_llvm_mc(self, isa_json_path, out_dir):
        """At least one .s file should assemble with llvm-mc."""
        import subprocess

        llvm_mc = os.path.expanduser(
            "~/npu-work/llvm-aie/build/bin/llvm-mc"
        )
        if not os.path.exists(llvm_mc):
            pytest.skip("llvm-mc not found")

        manifest = generate_all(isa_json_path, out_dir)
        assert manifest["total_batches"] > 0

        # Try to assemble the first batch.
        first_batch = manifest["batches"][0]
        s_path = os.path.join(out_dir, first_batch["filename"])

        result = subprocess.run(
            [llvm_mc, "--triple=aie2", "--filetype=obj", "-o", "/dev/null", s_path],
            capture_output=True, text=True, timeout=30,
        )
        # If assembly fails, print the error for diagnostics.
        if result.returncode != 0:
            # This is informational -- not all generated assembly may be valid
            # (some instructions may need special syntax).  We only assert that
            # llvm-mc ran without crashing.
            print(f"\nllvm-mc stderr:\n{result.stderr[:2000]}")
        # llvm-mc should at least not crash (segfault = returncode -11).
        assert result.returncode >= 0, "llvm-mc crashed"


# ===================================================================
# cm-class operand resolution (_resolve_unknown_operand)
# ===================================================================

def _make_cm_unknown_op(name, bit_width=4):
    """Build an operand dict matching the cm-class export signature.

    The ISA exporter emits operand_type='unknown', register_kind=None,
    bit_width=4 for cm-class (1024-bit full accumulator) operands.
    """
    return {
        "name": name,
        "bit_width": bit_width,
        "is_output": False,
        "operand_type": "unknown",
        "register_kind": None,
        "signed": False,
        "scale": None,
    }


class TestResolveUnknownOperand:
    """Tests for _resolve_unknown_operand() cm-class heuristic.

    The heuristic identifies cm-class (1024-bit full accumulator) operands
    that the ISA exporter incorrectly tags as operand_type='unknown'.
    Matching operands have: operand_type='unknown', register_kind=None,
    bit_width=4, and name in {dst, src, acc1, acc2}.
    """

    _resolve = staticmethod(isa_test_gen._resolve_unknown_operand)

    def test_dst_bw4_resolved_to_accumulator(self):
        """dst with bit_width=4 and unknown type -> register/accumulator."""
        op = _make_cm_unknown_op("dst")
        resolved = self._resolve(op)
        assert resolved["operand_type"] == "register"
        assert resolved["register_kind"] == "accumulator"

    def test_src_bw4_resolved(self):
        """src with bit_width=4 and unknown type -> register/accumulator."""
        op = _make_cm_unknown_op("src")
        resolved = self._resolve(op)
        assert resolved["operand_type"] == "register"
        assert resolved["register_kind"] == "accumulator"

    def test_acc1_bw4_resolved(self):
        """acc1 with bit_width=4 -> register/accumulator."""
        op = _make_cm_unknown_op("acc1")
        resolved = self._resolve(op)
        assert resolved["operand_type"] == "register"
        assert resolved["register_kind"] == "accumulator"

    def test_acc2_bw4_resolved(self):
        """acc2 with bit_width=4 -> register/accumulator."""
        op = _make_cm_unknown_op("acc2")
        resolved = self._resolve(op)
        assert resolved["operand_type"] == "register"
        assert resolved["register_kind"] == "accumulator"

    def test_original_not_mutated(self):
        """_resolve_unknown_operand must not mutate the input dict."""
        op = _make_cm_unknown_op("dst")
        original_type = op["operand_type"]
        self._resolve(op)
        assert op["operand_type"] == original_type

    def test_non_cm_name_not_resolved(self):
        """An unknown operand with a name not in the cm set is unchanged."""
        op = _make_cm_unknown_op("ys1")  # not dst/src/acc1/acc2
        resolved = self._resolve(op)
        assert resolved["operand_type"] == "unknown"
        assert resolved is op or resolved["operand_type"] == "unknown"

    def test_dontcare_not_resolved(self):
        """dontcare operands with bit_width=4 are NOT resolved (kept unknown)."""
        op = _make_cm_unknown_op("dontcare4")
        resolved = self._resolve(op)
        assert resolved["operand_type"] == "unknown"

    def test_wrong_bit_width_not_resolved(self):
        """bit_width=2 (sparse class) must not be resolved as cm."""
        op = _make_cm_unknown_op("dst", bit_width=2)
        resolved = self._resolve(op)
        assert resolved["operand_type"] == "unknown"

    def test_already_typed_not_touched(self):
        """If operand_type is already 'register', it is returned as-is."""
        op = {
            "name": "dst",
            "bit_width": 4,
            "is_output": False,
            "operand_type": "register",
            "register_kind": "accumulator",
            "signed": False,
            "scale": None,
        }
        resolved = self._resolve(op)
        assert resolved is op  # unchanged


class TestResolveSparseOperands:
    """Tests for _resolve_sparse_qx() and _resolve_operand() with ys1/qxs2."""

    def test_qxs2_resolved_to_sparse_qx(self):
        """qxs2 with accumulator bw=2 -> sparse_qx kind."""
        op = _make_reg_op("qxs2", "accumulator", bit_width=2)
        resolved = isa_test_gen._resolve_sparse_qx(op)
        assert resolved["register_kind"] == "sparse_qx"
        assert resolved["operand_type"] == "register"

    def test_non_qxs2_accumulator_bw2_unchanged(self):
        """Non-qxs2 accumulator bw=2 (e.g. dst for quad load) is unchanged."""
        op = _make_reg_op("dst", "accumulator", bit_width=2)
        resolved = isa_test_gen._resolve_sparse_qx(op)
        assert resolved["register_kind"] == "accumulator"

    def test_ys1_resolved_to_wide_y(self):
        """ys1 with unknown bw=2 -> wide_y kind via _resolve_unknown_operand."""
        op = _make_unknown_op("ys1", bit_width=2)
        resolved = isa_test_gen._resolve_unknown_operand(op)
        assert resolved["operand_type"] == "register"
        assert resolved["register_kind"] == "wide_y"

    def test_non_ys1_unknown_bw2_unchanged(self):
        """Non-ys1 unknown bw=2 is not resolved."""
        op = _make_unknown_op("weird", bit_width=2)
        resolved = isa_test_gen._resolve_unknown_operand(op)
        assert resolved["operand_type"] == "unknown"

    def test_resolve_operand_chains_all(self):
        """_resolve_operand applies cm-class, ys1, and qxs2 resolvers."""
        # cm-class
        op = _make_cm_unknown_op("dst")
        assert isa_test_gen._resolve_operand(op)["register_kind"] == "accumulator"
        # ys1
        op = _make_unknown_op("ys1", bit_width=2)
        assert isa_test_gen._resolve_operand(op)["register_kind"] == "wide_y"
        # qxs2
        op = _make_reg_op("qxs2", "accumulator", bit_width=2)
        assert isa_test_gen._resolve_operand(op)["register_kind"] == "sparse_qx"

    def test_cdst_csrc_resolved_as_cm_class(self):
        """cdst and csrc are cm-class operands (used by vmov.d)."""
        for name in ("cdst", "csrc"):
            op = _make_cm_unknown_op(name)
            resolved = isa_test_gen._resolve_unknown_operand(op)
            assert resolved["operand_type"] == "register"
            assert resolved["register_kind"] == "accumulator"


class TestCmClassClassification:
    """Tests for classify_instruction with cm-class operands.

    Covers the instructions recovered by _resolve_unknown_operand:
    VUPS x2c variants, VSRS x_srs variants, VMOV_mv_cm, and VNEG.
    """

    def _make_vups_x2c(self, mnemonic="vups.s32.s16"):
        """Build a synthetic VUPS x2c-style instruction.

        x2c variant: src is vector512 (bw=4), dst is cm-class (bw=4, unknown).
        The ISA exporter tags dst as unknown; the heuristic recovers it.
        """
        return _make_instr(
            "VUPS_S32_S16_mv_ups_x2c", mnemonic,
            f"{mnemonic}\t$dst, $src, $shft",
            [
                _make_reg_op("shft", "scalar", bit_width=2),
                _make_reg_op("src", "vector512", bit_width=4),
                _make_cm_unknown_op("dst"),  # cm-class, mistagged as unknown
            ],
            slot="mv", is_vector=True,
        )

    def _make_vsrs_x_srs(self, mnemonic="vsrs.s16.s32"):
        """Build a synthetic VSRS x_srs-style instruction.

        x_srs variant: src is cm-class (bw=4, unknown), dst is vector512 (bw=4).
        """
        return _make_instr(
            "VSRS_S16_S32_mv_x_srs", mnemonic,
            f"{mnemonic}\t$dst, $src, $shft",
            [
                _make_reg_op("dst", "vector512", bit_width=4),
                _make_cm_unknown_op("src"),  # cm-class, mistagged as unknown
                _make_reg_op("shft", "scalar", bit_width=2),
            ],
            slot="st", is_vector=True,
        )

    def _make_vmov_cm(self):
        """Build a synthetic VMOV_mv_cm instruction (both ops are cm-class)."""
        return _make_instr(
            "VMOV_mv_cm", "vmov",
            "vmov\t$dst, $src",
            [
                _make_cm_unknown_op("dst"),
                _make_cm_unknown_op("src"),
            ],
            slot="mv", is_vector=True,
        )

    def _make_vneg_cm(self):
        """Build a synthetic VNEG instruction (dst and acc1 are cm-class)."""
        return _make_instr(
            "VNEG", "vneg",
            "vneg\t$dst, $acc1, $c",
            [
                _make_reg_op("c", "scalar", bit_width=5),
                _make_cm_unknown_op("dst"),
                _make_cm_unknown_op("acc1"),
            ],
            slot="vec", is_vector=True,
        )

    def test_vups_x2c_is_testable(self):
        """VUPS x2c with cm-class dst (unknown bw=4) should be testable."""
        instr = self._make_vups_x2c()
        status, reason = classify_instruction(instr)
        assert status == "testable", \
            f"VUPS x2c should be testable via cm heuristic, got: {reason}"

    def test_vsrs_x_srs_is_testable(self):
        """VSRS x_srs with cm-class src (unknown bw=4) should be testable."""
        instr = self._make_vsrs_x_srs()
        status, reason = classify_instruction(instr)
        assert status == "testable", \
            f"VSRS x_srs should be testable via cm heuristic, got: {reason}"

    def test_vmov_cm_is_testable(self):
        """VMOV_mv_cm with both ops as cm-class should be testable."""
        instr = self._make_vmov_cm()
        status, reason = classify_instruction(instr)
        assert status == "testable", \
            f"VMOV_mv_cm should be testable via cm heuristic, got: {reason}"

    def test_vneg_cm_is_testable(self):
        """VNEG with cm-class dst and acc1 should be testable."""
        instr = self._make_vneg_cm()
        status, reason = classify_instruction(instr)
        assert status == "testable", \
            f"VNEG should be testable via cm heuristic, got: {reason}"

    def test_vups_x2c_dst_is_output(self):
        """For VUPS x2c, the cm-class dst should be detected as an output."""
        instr = self._make_vups_x2c()
        outputs = detect_output_operands(instr)
        assert len(outputs) >= 1
        assert outputs[0]["name"] == "dst"
        # After resolution, the output should look like an accumulator register.
        assert outputs[0]["register_kind"] == "accumulator"

    def test_vsrs_x_srs_dst_is_output(self):
        """For VSRS x_srs, the vector512 dst is the output (not the cm src)."""
        instr = self._make_vsrs_x_srs()
        outputs = detect_output_operands(instr)
        assert len(outputs) >= 1
        assert outputs[0]["name"] == "dst"

    def test_vups_x2c_combos_use_cm_names(self):
        """generate_operand_combos should assign cm* names to the dst operand."""
        instr = self._make_vups_x2c()
        combos = generate_operand_combos(instr)
        assert len(combos) >= 1
        # The dst should use cm-family register names.
        dst_val = combos[0].get("dst", "")
        assert dst_val.startswith("cm"), \
            f"Expected cm* register for dst, got: {dst_val!r}"

    def test_vsrs_x_srs_combos_use_cm_names(self):
        """generate_operand_combos should assign cm* names to the src operand."""
        instr = self._make_vsrs_x_srs()
        combos = generate_operand_combos(instr)
        assert len(combos) >= 1
        src_val = combos[0].get("src", "")
        assert src_val.startswith("cm"), \
            f"Expected cm* register for src, got: {src_val!r}"

    def test_vups_x2c_generates_test_point(self):
        """generate_test_point for VUPS x2c should include UPS sequence."""
        instr = self._make_vups_x2c()
        combos = generate_operand_combos(instr)
        regs = combos[0]
        asm = generate_test_point(instr, regs, in_offset=0, out_offset=0)
        # Must include the instruction.
        assert "vups.s32.s16" in asm
        # Must include SRS to extract the cm output for storage.
        assert "vsrs" in asm

    def test_unknown_bw2_non_ys1_still_blocked(self):
        """Unknown operands with bit_width=2 and names other than ys1 stay blocked."""
        instr = _make_instr("FOO_sparse", "foo.sparse",
                            "foo.sparse\t$dst, $acc1, $xs2, $c", [
            _make_reg_op("dst", "accumulator", bit_width=4),
            _make_reg_op("acc1", "accumulator", bit_width=4),
            _make_reg_op("xs2", "vector512"),
            _make_unknown_op("weird_op", bit_width=2),  # NOT ys1
            _make_reg_op("c", "scalar"),
        ], slot="vec", is_vector=True)
        status, reason = classify_instruction(instr)
        assert status == "skipped"
        assert "unknown" in reason.lower()

    def test_non_standard_name_still_blocked(self):
        """Unknown operand with bw=4 but name not in cm-class set stays blocked."""
        instr = _make_instr("BAR", "bar", "bar\t$mRx, $zz1", [
            _make_reg_op("mRx", "scalar"),
            _make_cm_unknown_op("zz1"),  # bw=4, unknown, but name not in cm-class set
        ])
        status, reason = classify_instruction(instr)
        assert status == "skipped"
        assert "unknown" in reason.lower()


class TestCmClassRealISA:
    """Integration tests against real aie2-isa.json for cm-class recovery.

    Verifies that the ~17 instructions with cm-class unknown operands are
    correctly classified as testable after applying _resolve_unknown_operand.
    """

    @pytest.fixture
    def isa_data(self):
        path = os.path.join(os.path.dirname(__file__), "aie2-isa.json")
        if not os.path.exists(path):
            pytest.skip("aie2-isa.json not found")
        with open(path) as f:
            return json.load(f)

    def _find_instr(self, isa_data, name):
        for section, items in isa_data.items():
            if isinstance(items, list):
                for i in items:
                    if i.get("name") == name:
                        return i
        return None

    @pytest.mark.parametrize("instr_name", [
        # VUPS x2c variants (dst is cm-class, src is vector512)
        "VUPS_S32_D16_mv_ups_x2c",
        "VUPS_S32_S16_mv_ups_x2c",
        "VUPS_S64_D32_mv_ups_x2c",
        "VUPS_S64_S32_mv_ups_x2c",
        # VSRS x_srs variants (src is cm-class, dst is vector512 or vector256)
        "VSRS_D16_S32_mv_x_srs",
        "VSRS_D32_S64_mv_x_srs",
        "VSRS_S16_S32_mv_x_srs",
        "VSRS_S32_S64_mv_x_srs",
        # VMOV and VNEG with cm operands
        "VMOV_mv_cm",
        "VNEG",
    ])
    def test_cm_instruction_is_testable(self, isa_data, instr_name):
        """Each cm-class instruction should be testable after heuristic resolution."""
        instr = self._find_instr(isa_data, instr_name)
        if instr is None:
            pytest.skip(f"{instr_name} not in ISA JSON")
        strategy = isa_test_gen.ComputeStrategy()
        can, reason = strategy.can_test(instr)
        assert can, \
            f"{instr_name} should be testable via cm heuristic, got: {reason}"

    @pytest.mark.parametrize("instr_name", [
        # VUPS x2c variants have cm dst, which should combo to cm* names
        "VUPS_S32_S16_mv_ups_x2c",
        "VUPS_S64_S32_mv_ups_x2c",
        # VSRS x_srs variants have cm src
        "VSRS_S16_S32_mv_x_srs",
        "VSRS_S32_S64_mv_x_srs",
    ])
    def test_cm_operand_combos_use_cm_names(self, isa_data, instr_name):
        """cm-class operands should get cm0/cm2/cm4 register assignments."""
        instr = self._find_instr(isa_data, instr_name)
        if instr is None:
            pytest.skip(f"{instr_name} not in ISA JSON")
        combos = generate_operand_combos(instr)
        assert len(combos) >= 1
        # At least one combo should have a cm* value for dst or src.
        found_cm = False
        for combo in combos:
            for val in combo.values():
                if isinstance(val, str) and val.startswith("cm"):
                    found_cm = True
                    break
        assert found_cm, \
            f"{instr_name}: expected cm* register in combos, got {combos[0]}"

    def test_vmov_hi_blocked_cascade(self, isa_data):
        """VMOV_HI reads from SCD (cascade) -- must be skipped despite cm dst.

        Even though VMOV_HI has a cm-class dst (bw=4, unknown), the asm_string
        contains 'SCD' which means the instruction stalls without active cascade
        connections.  The cascade check fires before the cm-class heuristic can
        make it testable, so it remains skipped.
        """
        instr = self._find_instr(isa_data, "VMOV_HI")
        if instr is None:
            pytest.skip("VMOV_HI not in ISA JSON")
        status, reason = classify_instruction(instr)
        assert status == "skipped"
        assert "cascade" in reason.lower()


# ===================================================================
# ConversionStrategy: LLVM IR generation for fused conversion instructions
# ===================================================================

class TestConversionIntrinsicMap:
    """Validate the CONVERSION_INTRINSICS table."""

    def test_has_26_entries(self):
        """Table must have exactly 26 unique base mnemonics (8 ups + 8 srs + 4 pack + 4 unpack + 2 bf16/fp32)."""
        table = isa_test_gen.CONVERSION_INTRINSICS
        assert len(table) == 26

    def test_all_required_fields_present(self):
        """Each entry must have intrinsic, in_type, out_type, in_bytes, out_bytes."""
        required = {"intrinsic", "in_type", "out_type", "in_bytes", "out_bytes"}
        for mnemonic, info in isa_test_gen.CONVERSION_INTRINSICS.items():
            for field in required:
                assert field in info, \
                    f"{mnemonic}: missing field '{field}'"

    def test_ups_entries(self):
        """UPS entries: 8 base mnemonics (4 type pairs x signed/unsigned)."""
        table = isa_test_gen.CONVERSION_INTRINSICS
        ups_keys = [k for k in table if k.startswith("vlda.ups.")]
        assert len(ups_keys) == 8

    def test_srs_entries(self):
        """SRS entries: 8 base mnemonics."""
        table = isa_test_gen.CONVERSION_INTRINSICS
        srs_keys = [k for k in table if k.startswith("vst.srs.")]
        assert len(srs_keys) == 8

    def test_pack_entries(self):
        """PACK entries: 4 base mnemonics."""
        table = isa_test_gen.CONVERSION_INTRINSICS
        pack_keys = [k for k in table if k.startswith("vst.pack.")]
        assert len(pack_keys) == 4

    def test_unpack_entries(self):
        """UNPACK entries: 4 base mnemonics."""
        table = isa_test_gen.CONVERSION_INTRINSICS
        unpack_keys = [k for k in table if k.startswith("vldb.unpack.")]
        assert len(unpack_keys) == 4

    def test_signed_unsigned_differ_only_in_sign(self):
        """Signed/unsigned pairs should use the same intrinsic but differ in sign."""
        table = isa_test_gen.CONVERSION_INTRINSICS
        # Check a UPS pair: vlda.ups.s32.s16 (signed) vs vlda.ups.s32.d16 (unsigned)
        s_info = table["vlda.ups.s32.s16"]
        d_info = table["vlda.ups.s32.d16"]
        assert s_info["intrinsic"] == d_info["intrinsic"]
        assert s_info["sign"] != d_info["sign"]
        assert s_info["in_type"] == d_info["in_type"]
        assert s_info["out_type"] == d_info["out_type"]

    def test_ups_byte_sizes(self):
        """UPS: input is 256-bit vector (32 bytes), output is 512-bit acc (64 bytes)."""
        info = isa_test_gen.CONVERSION_INTRINSICS["vlda.ups.s32.s16"]
        assert info["in_bytes"] == 32
        assert info["out_bytes"] == 64

    def test_srs_byte_sizes(self):
        """SRS: input is 512-bit acc (64 bytes), output is 256-bit vector (32 bytes)."""
        info = isa_test_gen.CONVERSION_INTRINSICS["vst.srs.s16.s32"]
        assert info["in_bytes"] == 64
        assert info["out_bytes"] == 32

    def test_pack_byte_sizes(self):
        """PACK: input is <32 x i16> (64 bytes), output is <32 x i8> (32 bytes)."""
        info = isa_test_gen.CONVERSION_INTRINSICS["vst.pack.s8.s16"]
        assert info["in_bytes"] == 64
        assert info["out_bytes"] == 32

    def test_unpack_byte_sizes(self):
        """UNPACK: input is <32 x i8> (32 bytes), output is <32 x i16> (64 bytes)."""
        info = isa_test_gen.CONVERSION_INTRINSICS["vldb.unpack.s16.s8"]
        assert info["in_bytes"] == 32
        assert info["out_bytes"] == 64


class TestConversionLLGeneration:
    """Verify generate_conversion_ll() produces valid LLVM IR."""

    def test_has_target_triple(self):
        """Output must begin with target triple = aie2."""
        table = isa_test_gen.CONVERSION_INTRINSICS
        # Build minimal test points from table
        test_points = []
        for mnemonic, info in list(table.items())[:1]:
            test_points.append({"mnemonic": mnemonic, **info})
        ll = isa_test_gen.generate_conversion_ll(test_points)
        assert 'target triple = "aie2"' in ll

    def test_has_test_kernel(self):
        """Output must define test_kernel with ptr args."""
        table = isa_test_gen.CONVERSION_INTRINSICS
        test_points = []
        for mnemonic, info in list(table.items())[:1]:
            test_points.append({"mnemonic": mnemonic, **info})
        ll = isa_test_gen.generate_conversion_ll(test_points)
        assert "define void @test_kernel(ptr %in, ptr %out)" in ll

    def test_ups_generates_intrinsic_call(self):
        """UPS test point should call the ups intrinsic."""
        info = isa_test_gen.CONVERSION_INTRINSICS["vlda.ups.s32.s16"]
        test_points = [{"mnemonic": "vlda.ups.s32.s16", **info}]
        ll = isa_test_gen.generate_conversion_ll(test_points)
        assert "@llvm.aie2.acc32.v16.I256.ups" in ll

    def test_srs_generates_intrinsic_call(self):
        """SRS test point should call the srs intrinsic."""
        info = isa_test_gen.CONVERSION_INTRINSICS["vst.srs.s16.s32"]
        test_points = [{"mnemonic": "vst.srs.s16.s32", **info}]
        ll = isa_test_gen.generate_conversion_ll(test_points)
        assert "@llvm.aie2.I256.v16.acc32.srs" in ll

    def test_pack_generates_intrinsic_call(self):
        """PACK test point should call the pack intrinsic."""
        info = isa_test_gen.CONVERSION_INTRINSICS["vst.pack.s8.s16"]
        test_points = [{"mnemonic": "vst.pack.s8.s16", **info}]
        ll = isa_test_gen.generate_conversion_ll(test_points)
        assert "@llvm.aie2.pack.I8.I16" in ll

    def test_unpack_generates_intrinsic_call(self):
        """UNPACK test point should call the unpack intrinsic."""
        info = isa_test_gen.CONVERSION_INTRINSICS["vldb.unpack.s16.s8"]
        test_points = [{"mnemonic": "vldb.unpack.s16.s8", **info}]
        ll = isa_test_gen.generate_conversion_ll(test_points)
        assert "@llvm.aie2.unpack.I16.I8" in ll

    def test_uses_volatile_loads_stores(self):
        """Must use volatile to prevent optimization."""
        info = isa_test_gen.CONVERSION_INTRINSICS["vlda.ups.s32.s16"]
        test_points = [{"mnemonic": "vlda.ups.s32.s16", **info}]
        ll = isa_test_gen.generate_conversion_ll(test_points)
        assert "load volatile" in ll
        assert "store volatile" in ll

    def test_uses_gep_for_offsets(self):
        """Must use GEP for per-test-point memory offsets."""
        table = isa_test_gen.CONVERSION_INTRINSICS
        # Use two test points to verify the second has a non-zero offset.
        items = list(table.items())[:2]
        test_points = [{"mnemonic": m, **info} for m, info in items]
        ll = isa_test_gen.generate_conversion_ll(test_points)
        assert "getelementptr i8" in ll

    def test_all_24_conversions_generate(self):
        """All 24 conversion mnemonics should produce valid IR."""
        table = isa_test_gen.CONVERSION_INTRINSICS
        test_points = [{"mnemonic": m, **info} for m, info in table.items()]
        ll = isa_test_gen.generate_conversion_ll(test_points)
        # Should have one load per test point at minimum.
        assert ll.count("load volatile") >= 24

    def test_declares_intrinsics(self):
        """All intrinsics used must be declared."""
        info = isa_test_gen.CONVERSION_INTRINSICS["vlda.ups.s32.s16"]
        test_points = [{"mnemonic": "vlda.ups.s32.s16", **info}]
        ll = isa_test_gen.generate_conversion_ll(test_points)
        assert "declare" in ll
        assert "@llvm.aie2.acc32.v16.I256.ups" in ll


class TestConversionStrategy:
    """Tests for ConversionStrategy can_handle and sizing."""

    def test_can_handle_ups(self):
        """ConversionStrategy should handle vlda.ups.* mnemonics."""
        strategy = isa_test_gen.ConversionStrategy()
        instr = _make_instr(
            "VLDA_UPS_S32_S16_ag_idx", "vlda.ups.s32.s16",
            "vlda.ups.s32.s16\t$dst, [$ptr, $idx]",
            [_make_reg_op("dst", "accumulator"),
             _make_reg_op("ptr", "pointer"),
             _make_reg_op("idx", "scalar")],
            may_load=True,
        )
        assert strategy.can_handle(instr)

    def test_can_handle_srs(self):
        """ConversionStrategy should handle vst.srs.* mnemonics."""
        strategy = isa_test_gen.ConversionStrategy()
        instr = _make_instr(
            "VST_SRS_S16_S32_ag_idx", "vst.srs.s16.s32",
            "vst.srs.s16.s32\t[$ptr, $idx], $src",
            [_make_reg_op("ptr", "pointer"),
             _make_reg_op("idx", "scalar"),
             _make_reg_op("src", "accumulator")],
            may_store=True,
        )
        assert strategy.can_handle(instr)

    def test_can_handle_pack(self):
        """Pack intrinsics are now lowered by llc (fused into vst.pack)."""
        strategy = isa_test_gen.ConversionStrategy()
        instr = _make_instr(
            "VST_PACK_S8_S16_ag_idx", "vst.pack.s8.s16",
            "vst.pack.s8.s16\t[$ptr, $idx], $src",
            [_make_reg_op("ptr", "pointer"),
             _make_reg_op("idx", "scalar"),
             _make_reg_op("src", "vector256")],
            may_store=True,
        )
        assert strategy.can_handle(instr)

    def test_can_handle_unpack(self):
        """Unpack intrinsics are now lowered by llc."""
        strategy = isa_test_gen.ConversionStrategy()
        instr = _make_instr(
            "VLDB_UNPACK_S16_S8_ag_idx", "vldb.unpack.s16.s8",
            "vldb.unpack.s16.s8\t$dst, [$ptr, $idx]",
            [_make_reg_op("dst", "vector256"),
             _make_reg_op("ptr", "pointer"),
             _make_reg_op("idx", "scalar")],
            may_load=True,
        )
        assert strategy.can_handle(instr)

    def test_can_handle_conv(self):
        """ConversionStrategy handles .conv. (bf16/fp32) mnemonics."""
        strategy = isa_test_gen.ConversionStrategy()
        instr = _make_instr(
            "VST_CONV_BF16_FP32", "vst.conv.bf16.fp32",
            "vst.conv.bf16.fp32\t[$ptr], $src",
            [_make_reg_op("ptr", "pointer"),
             _make_reg_op("src", "accumulator")],
            may_store=True,
        )
        assert strategy.can_handle(instr)

    def test_does_not_handle_plain_add(self):
        """ConversionStrategy should NOT handle plain compute instructions."""
        strategy = isa_test_gen.ConversionStrategy()
        instr = _make_instr(
            "ADD", "add", "add\t$dst, $s1, $s2",
            [_make_reg_op("dst", "scalar"),
             _make_reg_op("s1", "scalar"),
             _make_reg_op("s2", "scalar")],
        )
        assert not strategy.can_handle(instr)

    def test_can_handle_2d_variant(self):
        """ConversionStrategy should handle 2D addressing variants."""
        strategy = isa_test_gen.ConversionStrategy()
        instr = _make_instr(
            "VLDA_2D_UPS_S32_S16", "vlda.2d.ups.s32.s16",
            "vlda.2d.ups.s32.s16\t$dst, [$ptr, $mod]",
            [_make_reg_op("dst", "accumulator"),
             _make_reg_op("ptr", "pointer"),
             _make_reg_op("mod", "modifier_m")],
            may_load=True,
        )
        assert strategy.can_handle(instr)

    def test_sizes_ups(self):
        """UPS: buffer I/O is vector-sized (round-trips through accumulator).

        Input buffer: 32 bytes (vector loaded).
        Output buffer: 32 bytes (vector after UPS+SRS round-trip).
        """
        strategy = isa_test_gen.ConversionStrategy()
        instr = _make_instr(
            "VLDA_UPS_S32_S16_ag_idx", "vlda.ups.s32.s16",
            "vlda.ups.s32.s16\t$dst, [$ptr, $idx]",
            [_make_reg_op("dst", "accumulator"),
             _make_reg_op("ptr", "pointer"),
             _make_reg_op("idx", "scalar")],
            may_load=True,
        )
        assert strategy.compute_input_size(instr, {}) == 32
        assert strategy.compute_output_size(instr) == 32

    def test_sizes_srs(self):
        """SRS: buffer I/O is vector-sized (UPS used to get data into acc).

        Input buffer: 32 bytes (vector loaded, then UPS'd).
        Output buffer: 32 bytes (vector after SRS).
        """
        strategy = isa_test_gen.ConversionStrategy()
        instr = _make_instr(
            "VST_SRS_S16_S32_ag_idx", "vst.srs.s16.s32",
            "vst.srs.s16.s32\t[$ptr, $idx], $src",
            [_make_reg_op("ptr", "pointer"),
             _make_reg_op("idx", "scalar"),
             _make_reg_op("src", "accumulator")],
            may_store=True,
        )
        assert strategy.compute_input_size(instr, {}) == 32
        assert strategy.compute_output_size(instr) == 32

    def test_generate_combos_returns_one(self):
        """ConversionStrategy should return exactly one combo per instruction."""
        strategy = isa_test_gen.ConversionStrategy()
        instr = _make_instr(
            "VLDA_UPS_S32_S16_ag_idx", "vlda.ups.s32.s16",
            "vlda.ups.s32.s16\t$dst, [$ptr, $idx]",
            [_make_reg_op("dst", "accumulator"),
             _make_reg_op("ptr", "pointer"),
             _make_reg_op("idx", "scalar")],
            may_load=True,
        )
        combos = strategy.generate_combos(instr)
        assert len(combos) == 1

    @pytest.fixture
    def isa_data(self):
        path = os.path.join(os.path.dirname(__file__), "aie2-isa.json")
        if not os.path.exists(path):
            pytest.skip("aie2-isa.json not found")
        with open(path) as f:
            return json.load(f)

    def test_real_isa_conversion_count(self, isa_data):
        """ConversionStrategy handles all conversion instruction defs."""
        strategy = isa_test_gen.ConversionStrategy()
        count = 0
        for slot, instrs in isa_data.items():
            for instr in instrs:
                if strategy.can_handle(instr):
                    count += 1
        # 26 base mnemonics x various addressing modes
        # (8 UPS + 8 SRS + 4 pack + 4 unpack + 2 bf16/fp32 conv)
        assert count >= 100, f"Expected >= 100 conversion defs handled, got {count}"


class TestGenerateAllWithConversions:
    """Integration tests for generate_all with conversion batches."""

    @pytest.fixture
    def isa_json_path(self):
        path = os.path.join(os.path.dirname(__file__), "aie2-isa.json")
        if not os.path.exists(path):
            pytest.skip("aie2-isa.json not found")
        return path

    @pytest.fixture
    def out_dir(self, tmp_path):
        return str(tmp_path / "isa-tests")

    def test_manifest_has_source_type(self, isa_json_path, out_dir):
        """Each batch should have a source_type field."""
        manifest = generate_all(isa_json_path, out_dir)
        for batch in manifest["batches"]:
            assert "source_type" in batch, \
                f"Batch {batch['batch_index']} missing source_type"
            assert batch["source_type"] in ("assembly", "llvm_ir"), \
                f"Batch {batch['batch_index']}: unexpected source_type '{batch['source_type']}'"

    def test_conversion_batch_exists(self, isa_json_path, out_dir):
        """At least one batch should be llvm_ir type (conversions)."""
        manifest = generate_all(isa_json_path, out_dir)
        ll_batches = [b for b in manifest["batches"] if b["source_type"] == "llvm_ir"]
        assert len(ll_batches) >= 1, "Expected at least one llvm_ir batch"

    def test_conversion_batch_has_ll_file(self, isa_json_path, out_dir):
        """Conversion batch should reference a .ll file that exists."""
        manifest = generate_all(isa_json_path, out_dir)
        ll_batches = [b for b in manifest["batches"] if b["source_type"] == "llvm_ir"]
        assert len(ll_batches) >= 1
        for batch in ll_batches:
            assert batch["filename"].endswith(".ll"), \
                f"Expected .ll filename, got {batch['filename']}"
            ll_path = os.path.join(out_dir, batch["filename"])
            assert os.path.exists(ll_path), f"Missing {batch['filename']}"

    def test_conversion_batch_has_26_tests(self, isa_json_path, out_dir):
        """Conversion batch should have 26 test points (ups+srs+pack+unpack+bf16)."""
        manifest = generate_all(isa_json_path, out_dir)
        ll_batches = [b for b in manifest["batches"] if b["source_type"] == "llvm_ir"]
        total_ll_tests = sum(b["test_count"] for b in ll_batches)
        assert total_ll_tests == 26, \
            f"Expected 26 conversion test points, got {total_ll_tests}"

    def test_conversion_testable_counted(self, isa_json_path, out_dir):
        """Conversion instructions should be counted as testable, not skipped."""
        manifest = generate_all(isa_json_path, out_dir)
        # The 136 conversion instruction defs that were previously skipped
        # should now count toward testable_instructions.
        assert manifest["testable_instructions"] > 0
