"""The flag-clear race probe generator."""
import gen_flag_clear_probe as g


def test_the_clear_sits_n_bundles_after_the_conversion():
    form = g.FORMS[0]
    for n in (1, 3):
        lines = g.record_lines(form, n)
        op = next(i for i, l in enumerate(lines) if l.strip() == form.op)
        assert lines[op + n].strip() == f"mov {form.flag}, r0"
    assert any("; mov srSRS_of, r0" in l for l in g.record_lines(form, 0))
    assert g.record_lines(g.FORMS[2], 0) == []     # vups cannot share the mv slot


def test_the_store_scratch_is_vector_aligned_past_the_flag_words():
    assert g.SCRATCH % 32 == 0 and g.SCRATCH >= 4 * len(g.OFFSETS)
