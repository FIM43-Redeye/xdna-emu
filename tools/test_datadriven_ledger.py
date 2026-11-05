"""Tests for the schema-availability (M9) part of tools/datadriven_ledger.py."""
from datadriven_ledger import operand_type_agrees, schema_report


def reg(family, encoder=None):
    return {"register": {"class": "x", "family": family, "bits": 32, "encoder": encoder, "fixed": None}}


def imm(signed, step=1, negative=False, bits=4):
    return {"immediate": {"bits": bits, "step": step, "signed": signed, "negative": negative}}


def test_register_types_agree_by_family():
    assert operand_type_agrees("OperandType::Register(RegisterKind::Scalar)", reg("Scalar"))
    assert operand_type_agrees("OperandType::RegisterWithOffset(RegisterKind::Scalar, 16)", reg("Scalar"))
    assert operand_type_agrees("OperandType::Register(RegisterKind::Accumulator)", reg("Acc1024"))
    # Audit bug 2: q registers were classified as accumulators.
    assert not operand_type_agrees("OperandType::Register(RegisterKind::Accumulator)", reg("Vector128"))
    assert not operand_type_agrees("OperandType::Unknown", reg("Acc1024"))


def test_composite_and_immediate_types():
    assert operand_type_agrees("OperandType::CompositeRegister(CompositeEncoder::LdaScl)", reg("Scalar", "getmLdaSclOpValue"))
    assert not operand_type_agrees("OperandType::CompositeRegister(CompositeEncoder::LdaScl)", reg("Scalar"))
    assert operand_type_agrees("OperandType::Immediate { signed: true, scale: 4 }", imm(True, 4))
    # Audit bug 2: unsigned immediates marked signed, negative operands with scale 1.
    assert not operand_type_agrees("OperandType::Immediate { signed: true, scale: 1 }", imm(False))
    assert not operand_type_agrees("OperandType::Immediate { signed: true, scale: 1 }", imm(True, 32, True))
    assert operand_type_agrees("OperandType::LockId", imm(False))


def encoding(**over):
    e = {
        "name": "X", "may_load": True, "may_store": False, "addressing_mode": "AddressingMode::IndexedImmediate",
        "mem_width": "InstrMemWidth::HalfWord", "implicit_regs": [],
        "operand_fields": [{"name": "imm", "operand_type": "OperandType::Immediate { signed: true, scale: 1 }"},
                           {"name": "raw", "operand_type": "OperandType::Unknown"}],
        "schema": {"operands": [{"name": "imm", "kind": imm(True)}], "addressing": "IndexedImm",
                   "mem_vt_bits": 16, "implicit_regs": [], "format_class": "F"},
    }
    e.update(over)
    return e


def test_schema_report_counts_and_lists_disagreements():
    m9, dis = schema_report([encoding(), encoding(name="Y", addressing_mode="AddressingMode::Unknown")])
    assert m9["register_classes"] == {"applicable": 4, "schema": "2/4", "agree_with_legacy": "2/2"}
    assert m9["addressing_mode"]["agree_with_legacy"] == "1/2"
    assert m9["mem_width"] == {"applicable": 2, "schema": "2/2", "agree_with_legacy": "2/2"}
    assert dis == [("Y", "addressing_mode", "", "AddressingMode::Unknown", "IndexedImm")]
