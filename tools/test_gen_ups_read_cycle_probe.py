"""The vlda.ups read-cycle probe generator."""
import gen_ups_read_cycle_probe as g


def test_the_late_write_sits_n_bundles_after_the_load():
    form = g.FORMS[1]
    for n in (1, 6):
        lines = g.record_lines(form, "sat", n)
        load = next(i for i, l in enumerate(lines) if l.strip().startswith(form.mnemonic))
        assert lines[load + n].strip() == "movx crSat, r6"
    same = g.record_lines(form, "sign", 0)
    assert any(form.mnemonic in l and "; movx crUPSSign, r6" in l for l in same)


def test_every_form_sweeps_every_register_and_offset():
    assert len(g.records_of()) == len(g.PARAMS) * len(g.OFFSETS)
    for form in g.FORMS:
        text = g.kernel_text(form)
        assert text.count(form.mnemonic) == len(g.records_of())
        assert len(g.source_bytes(form)) == 32
