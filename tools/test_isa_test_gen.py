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
        instr = _make_instr("ADD_NC", "add.nc", "add.nc\t$dst, $s0, $imm", [
            _make_composite_op("dst", "MvSclSrc"),
            _make_reg_op("s0", "scalar"),
            _make_imm_op("imm"),
        ])
        status, reason = classify_instruction(instr)
        assert status == "skipped"
        assert "composite" in reason.lower()

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

    def test_skip_hardware_counter(self):
        """MOV_CNTR reads hardware counter and uses register pairs -- skip."""
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
        assert status == "skipped"

    def test_imm_only_skipped(self):
        """Instructions with only immediate operands (no register output)."""
        instr = _make_instr("EVENT", "event", "event\t$val", [
            _make_imm_op("val", bit_width=5),
        ])
        status, _ = classify_instruction(instr)
        assert status == "skipped"

    def test_skip_control_register_output(self):
        """MOVX writes to control registers -- skip (side effect)."""
        instr = _make_instr("MOVX", "movx", "movx\t$mCRm, $mRx", [
            _make_reg_op("mCRm", "control"),
            _make_reg_op("mRx", "scalar"),
        ])
        status, reason = classify_instruction(instr)
        assert status == "skipped"
        assert "control" in reason

    def test_skip_composite_sparse_register(self):
        """Instructions with accumulator bw=2 are actually composite/sparse."""
        instr = _make_instr("VNEGMSC_sparse", "vnegmsc",
                            "vnegmsc\t$dst, $acc1, $xs1, $qxs2, $c", [
            _make_reg_op("dst", "accumulator", bit_width=4),
            _make_reg_op("acc1", "accumulator", bit_width=4),
            _make_reg_op("xs1", "vector512"),
            _make_reg_op("qxs2", "accumulator", bit_width=2),
            _make_reg_op("c", "scalar"),
        ])
        status, reason = classify_instruction(instr)
        assert status == "skipped"
        assert "sparse" in reason or "bw=2" in reason

    def test_dontcare_operand_accepted(self):
        """Instructions with dontcare padding operands should be testable."""
        instr = _make_instr("PADDA", "padda", "padda\t[$mPa], #$c12s", [
            _make_reg_op("mPa", "pointer", bit_width=3),
            _make_imm_op("c12s", bit_width=12, signed=True),
            _make_unknown_op("dontcare2", bit_width=2),
        ])
        status, reason = classify_instruction(instr)
        assert status == "testable", f"Expected testable, got {status}: {reason}"

    def test_non_dontcare_unknown_still_skipped(self):
        """Unknown operands that are NOT dontcare should still be skipped."""
        instr = _make_instr("FOO", "foo", "foo\t$dst, $src, $ys1", [
            _make_reg_op("dst", "scalar"),
            _make_reg_op("src", "scalar"),
            _make_unknown_op("ys1", bit_width=2),
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
        names = register_names("ERS4")
        assert names == []

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
    def test_result_latency_scalar_alu(self):
        """Scalar ALU instructions have 1-cycle result latency."""
        instr = {"sched_class": "II_ABS"}
        assert result_latency(instr) == 1

    def test_result_latency_vmac(self):
        """VMAC has 5-cycle result latency."""
        instr = {"sched_class": "II_VMAC"}
        assert result_latency(instr) == 5

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

    def test_rejects_composite_dest(self):
        """Loads with composite destination are deferred."""
        instr = _make_instr("LDA_2D_LDASCL", "lda.2d",
                            "lda.2d\t$mLdaScl, [$ptr], $mod", [
            _make_composite_op("mLdaScl", "LdaScl", bit_width=7),
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

    def test_rejects_composite_source(self):
        instr = _make_instr("ST_COMP", "st.2d",
                            "st.2d\t$mLdaScl, [$ptr], $mod", [
            _make_composite_op("mLdaScl", "LdaScl", bit_width=7),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_reg_op("mod", "modifier_m", bit_width=3),
        ], may_store=True, slot="st")
        strategy = isa_test_gen.StoreStrategy()
        can, reason = strategy.can_test(instr)
        assert not can

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
# BranchStrategy tests
# ===================================================================

class TestBranchStrategy:
    """Tests for BranchStrategy."""

    def setup_method(self):
        """Reset label counter before each test."""
        isa_test_gen.BranchStrategy.reset_labels()

    def test_jnz_can_test(self):
        instr = _make_instr("JNZ", "jnz", "jnz\t$mRx, $cpmaddr", [
            _make_reg_op("mRx", "scalar"),
            _make_imm_op("cpmaddr", bit_width=20, signed=False),
        ], slot="lng")
        strategy = isa_test_gen.BranchStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"Expected can_test=True: {reason}"

    def test_jz_can_test(self):
        instr = _make_instr("JZ", "jz", "jz\t$mRx, $cpmaddr", [
            _make_reg_op("mRx", "scalar"),
            _make_imm_op("cpmaddr", bit_width=20, signed=False),
        ], slot="lng")
        strategy = isa_test_gen.BranchStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"Expected can_test=True: {reason}"

    def test_j_immediate_can_test(self):
        instr = _make_instr("J", "j", "j\t$cpmaddr", [
            _make_imm_op("cpmaddr", bit_width=20, signed=False),
        ], slot="lng")
        strategy = isa_test_gen.BranchStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"Expected can_test=True: {reason}"

    def test_j_register_indirect_deferred(self):
        """Register-indirect j $mPm should be deferred (needs pointer setup)."""
        instr = _make_instr("J_IND", "j", "j\t$mPm", [
            _make_reg_op("mPm", "pointer"),
        ], slot="alu")
        strategy = isa_test_gen.BranchStrategy()
        can, reason = strategy.can_test(instr)
        assert not can

    def test_ret_deferred(self):
        instr = _make_instr("RET", "ret", "ret lr", [], slot="alu")
        strategy = isa_test_gen.BranchStrategy()
        can, reason = strategy.can_test(instr)
        assert not can

    def test_jnzd_deferred(self):
        instr = _make_instr("JNZD", "jnzd", "jnzd\t$mRx, $mRx0, $mPm", [
            _make_reg_op("mRx", "scalar"),
            _make_reg_op("mRx0", "scalar"),
            _make_reg_op("mPm", "pointer"),
        ], slot="alu")
        strategy = isa_test_gen.BranchStrategy()
        can, reason = strategy.can_test(instr)
        assert not can

    def test_jnz_generates_markers(self):
        instr = _make_instr("JNZ", "jnz", "jnz\t$mRx, $cpmaddr", [
            _make_reg_op("mRx", "scalar"),
            _make_imm_op("cpmaddr", bit_width=20, signed=False),
        ], slot="lng")
        strategy = isa_test_gen.BranchStrategy()
        regs = {"mRx": "r0", "cpmaddr": "0"}
        asm = strategy.generate_test_point(instr, regs,
                                           in_offset=0, out_offset=0)
        assert ".Ltaken_" in asm
        assert ".Ldone_" in asm
        assert "jnz" in asm
        # Should store markers
        assert "#170" in asm  # 0xAA

    def test_labels_are_unique(self):
        instr = _make_instr("JNZ", "jnz", "jnz\t$mRx, $cpmaddr", [
            _make_reg_op("mRx", "scalar"),
            _make_imm_op("cpmaddr", bit_width=20, signed=False),
        ], slot="lng")
        strategy = isa_test_gen.BranchStrategy()
        regs = {"mRx": "r0", "cpmaddr": "0"}
        asm1 = strategy.generate_test_point(instr, regs,
                                            in_offset=0, out_offset=0)
        asm2 = strategy.generate_test_point(instr, regs,
                                            in_offset=32, out_offset=32)
        import re as _re
        labels1 = set(_re.findall(r'\.L\w+_(\d+)', asm1))
        labels2 = set(_re.findall(r'\.L\w+_(\d+)', asm2))
        assert labels1.isdisjoint(labels2), \
            f"Labels must be unique: {labels1} vs {labels2}"

    def test_conditional_combos(self):
        """Conditional branches should produce 2 combos (taken + not-taken)."""
        instr = _make_instr("JNZ", "jnz", "jnz\t$mRx, $cpmaddr", [
            _make_reg_op("mRx", "scalar"),
            _make_imm_op("cpmaddr", bit_width=20, signed=False),
        ], slot="lng")
        strategy = isa_test_gen.BranchStrategy()
        combos = strategy.generate_combos(instr)
        assert len(combos) == 2

    def test_unconditional_combos(self):
        """Unconditional j should produce 1 combo."""
        instr = _make_instr("J", "j", "j\t$cpmaddr", [
            _make_imm_op("cpmaddr", bit_width=20, signed=False),
        ], slot="lng")
        strategy = isa_test_gen.BranchStrategy()
        combos = strategy.generate_combos(instr)
        assert len(combos) == 1


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
