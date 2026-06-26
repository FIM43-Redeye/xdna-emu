from accuracy_baseline import (parse_divergence_score, harvest, render_markdown,
                               Entry)


def test_parse_divergence_score_sums_all_four_counts():
    line = ("TRACE_VERDICT: DIVERGE (edge: 3 diverged, 1 mismatch; "
            "level: 2 diverged, 4 mismatch)")
    assert parse_divergence_score(line) == 10


def test_parse_divergence_score_clean_is_zero():
    assert parse_divergence_score("TRACE_VERDICT: CLEAN") == 0


def _write_pair(d, kernel, compiler, verdict, log_line):
    (d / f"{kernel}.{compiler}.trace.summary").write_text(verdict + "\n")
    (d / f"{kernel}.{compiler}.trace.log").write_text(log_line + "\n")


def test_harvest_tallies_and_scores(tmp_path):
    _write_pair(tmp_path, "add_one_using_dma", "chess", "CLEAN",
                "TRACE_VERDICT: CLEAN")
    _write_pair(tmp_path, "vec_mul_distribute_lateral", "chess", "DIVERGE",
                "TRACE_VERDICT: DIVERGE (edge: 0 diverged, 0 mismatch; "
                "level: 5 diverged, 2 mismatch)")
    entries = harvest(str(tmp_path))
    by_key = {(e.kernel, e.compiler): e for e in entries}
    assert by_key[("add_one_using_dma", "chess")].verdict == "CLEAN"
    assert by_key[("add_one_using_dma", "chess")].score == 0
    assert by_key[("vec_mul_distribute_lateral", "chess")].score == 7


def test_harvest_marks_documented_gap(tmp_path):
    _write_pair(tmp_path, "vec_mul_trace_distribute_lateral", "chess", "DIVERGE",
                "TRACE_VERDICT: DIVERGE (edge: 1 diverged, 0 mismatch; "
                "level: 0 diverged, 0 mismatch)")
    kg = tmp_path / "known.md"
    kg.write_text("the residual is on vec_mul_trace_distribute_lateral ...")
    entries = harvest(str(tmp_path), known_gaps_path=str(kg))
    assert entries[0].documented is True


def test_render_markdown_has_tally_and_ranked_diverge(tmp_path):
    entries = [Entry("k_clean", "chess", "CLEAN", 0, False),
               Entry("k_big", "chess", "DIVERGE", 9, False),
               Entry("k_small", "peano", "DIVERGE", 2, True)]
    md = render_markdown(entries, "20260626")
    assert "1 CLEAN" in md and "2 DIVERGE" in md
    # ranked: the bigger divergence appears before the smaller one
    assert md.index("k_big") < md.index("k_small")


def test_render_markdown_multiple_clean_no_crash():
    # Two CLEAN entries: Entry has no __lt__, so sort without key= raises TypeError.
    # The key= fix must be present for this to pass.
    entries = [Entry("z_kernel", "peano", "CLEAN", 0, False),
               Entry("a_kernel", "chess", "CLEAN", 0, False)]
    md = render_markdown(entries, "20260626")
    assert "2 CLEAN" in md
    # Sorted alphabetically by (kernel, compiler): a_kernel before z_kernel
    assert md.index("a_kernel") < md.index("z_kernel")
