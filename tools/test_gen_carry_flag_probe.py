"""The carry-flag probe generator and its silicon model."""
import gen_carry_flag_probe as g


def test_records_cover_every_tablegen_definer_and_chains():
    ops = {r.op for r in g.records()}
    assert {"add", "sub", "adc", "sbc", "abs", "jnzd", "add_imm-1", "add_nc", "mul"} <= ops
    assert any(r.chain == "add@1" and r.op == "adc" for r in g.records())


def test_the_model_is_carry_out_for_add_and_borrow_for_sub():
    assert g.model("add", 0xFFFF_FFFF, 1, 0) == (0, 1)
    assert g.model("sub", 0, 1, 0) == (0xFFFF_FFFF, 1)
    assert g.model("sub", 2, 1, 1) == (1, 0)
    assert g.model("sbc", 1, 0, 1) == (0, 0)
    assert g.model("sbc", 0, 0x7FFF_FFFF, 1) == (0x8000_0000, 1)
    assert g.model("adc", 0, 0xFFFF_FFFF, 1) == (0, 1)
    assert g.model("jnzd", 0, 0, 1) == (0xFFFF_FFFF, 0)
    assert g.model("abs", 0x8000_0000, 0, 0) == (0x8000_0000, 1)
    assert g.model("add_imm-1", 5, 0, 0) == (4, 1)
    assert g.model("mul", 3, 4, 1)[1] is None


def test_kernel_text_reads_the_flag_in_the_bundle_and_after():
    text = g.kernel_text(g.records()[:3])
    assert "add r4, r2, r3 ; mov r8, srCarry" in text
    assert text.count("mov srCarry, r6") == 3
    assert "mov srCarry, r14" in text            # the caller's carry restored
