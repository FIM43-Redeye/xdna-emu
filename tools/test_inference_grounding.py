import json
from inference.grounding import (same_domain, ground_edge, assemble,
                                 Segment, Gap, Timeline, is_async_cdc,
                                 GAP_ASYNC_CDC, GAP_CROSS_DOMAIN,
                                 GAP_WITHIN_DOMAIN_NONEXACT, gap_accounted)


def _ev(col, row, name, soc, pkt_type=0):
    return {"col": col, "row": row, "pkt_type": pkt_type, "slot": 0,
            "name": name, "ts": soc, "soc": soc, "mode": 0}


def _runs(tmp_path, rows):
    """rows: list over runs of {key: anchored_offset}. Keys are full
    'col|row|pkt|name' strings so a test can place events in any domain."""
    dirs = []
    for i, row in enumerate(rows):
        rd = tmp_path / f"run{i}"
        evs = [_ev(1, 2, "PERF_CNT_2", 1000)]
        for key, delta in row.items():
            col, r, pkt, name = key.split("|")
            evs.append(_ev(int(col), int(r), name, 1000 + delta, pkt_type=int(pkt)))
        (rd / "batch_00" / "hw").mkdir(parents=True)
        (rd / "batch_00" / "hw" / "trace.events.json").write_text(
            json.dumps({"schema_version": 1, "events": evs, "slot_names": {}}))
        dirs.append(str(rd))
    return dirs


def test_same_domain_true_for_identical_col_row_pkt():
    assert same_domain("1|2|0|ACQUIRE", "1|2|0|RELEASE") is True


def test_same_domain_false_for_different_pkt_type():
    # same tile (1,2) but different module (pkt 0 vs 3) -> CROSS domain (C1)
    assert same_domain("1|2|0|A", "1|2|3|B") is False


def test_same_domain_false_for_different_tile():
    assert same_domain("1|2|0|A", "1|0|0|B") is False


def test_ground_edge_segment_when_same_domain_and_exact(tmp_path):
    dirs = _runs(tmp_path, [{"1|2|0|ACQ": 0, "1|2|0|REL": 22},
                            {"1|2|0|ACQ": 50, "1|2|0|REL": 72}])
    g = ground_edge(dirs, "1|2|0|REL", "1|2|0|ACQ")
    assert g == Segment(parent="1|2|0|ACQ", child="1|2|0|REL", offset=22)


def test_ground_edge_gap_when_same_domain_but_nonexact(tmp_path):
    dirs = _runs(tmp_path, [{"1|2|0|ACQ": 0, "1|2|0|REL": 22},
                            {"1|2|0|ACQ": 50, "1|2|0|REL": 73}])  # range 1
    g = ground_edge(dirs, "1|2|0|REL", "1|2|0|ACQ")
    assert g == Gap(parent="1|2|0|ACQ", child="1|2|0|REL",
                    reason=GAP_WITHIN_DOMAIN_NONEXACT)


def test_ground_edge_gap_when_cross_domain_even_if_exact(tmp_path):
    # exact offset (30 every run) but different modules (shim pkt 2 vs core pkt 0)
    # -> still a Gap (not a Segment), now carrying the reproduction offset.
    dirs = _runs(tmp_path, [{"1|0|2|MM2S": 0, "1|2|0|CORE": 30},
                            {"1|0|2|MM2S": 5, "1|2|0|CORE": 35}])
    g = ground_edge(dirs, "1|2|0|CORE", "1|0|2|MM2S")
    assert g == Gap(parent="1|0|2|MM2S", child="1|2|0|CORE",
                    reason=GAP_CROSS_DOMAIN, reproduction_offset=30)


def test_assemble_interleaves_segments_and_gaps(tmp_path):
    # chain: shim MM2S (parent) -> core ACQ -> core REL.
    # MM2S->ACQ is cross-domain (gap); ACQ->REL is within-domain exact (segment).
    dirs = _runs(tmp_path, [{"1|0|2|MM2S": 0, "1|2|0|ACQ": 30, "1|2|0|REL": 52},
                            {"1|0|2|MM2S": 7, "1|2|0|ACQ": 40, "1|2|0|REL": 62}])
    tl = assemble(dirs, [("1|0|2|MM2S", "1|2|0|ACQ"), ("1|2|0|ACQ", "1|2|0|REL")])
    assert isinstance(tl, Timeline)
    assert tl.items[0] == Gap(parent="1|0|2|MM2S", child="1|2|0|ACQ",
                              reason=GAP_CROSS_DOMAIN)
    assert tl.items[1] == Segment(parent="1|2|0|ACQ", child="1|2|0|REL", offset=22)


def test_gap_accounted_classifies_reasons():
    # structural non-exactness we understand -> accounted (NOTE);
    # a within-domain span that should be exact but isn't -> unaccounted (WARN).
    assert gap_accounted(GAP_ASYNC_CDC) is True
    assert gap_accounted(GAP_CROSS_DOMAIN) is True
    assert gap_accounted(GAP_WITHIN_DOMAIN_NONEXACT) is False


def test_within_domain_nonexact_gap_is_unaccounted(tmp_path):
    # within-domain span that SHOULD be cycle-exact but ranges -> the canary
    # anomaly: tagged unaccounted so the engine warns instead of swallowing it.
    dirs = _runs(tmp_path, [{"1|2|0|ACQ": 0, "1|2|0|REL": 22},
                            {"1|2|0|ACQ": 50, "1|2|0|REL": 73}])  # range 1
    g = ground_edge(dirs, "1|2|0|REL", "1|2|0|ACQ")
    assert g.reason == GAP_WITHIN_DOMAIN_NONEXACT
    assert g.accounted is False


def test_cross_domain_gap_is_accounted(tmp_path):
    dirs = _runs(tmp_path, [{"1|0|2|MM2S": 0, "1|2|0|CORE": 30},
                            {"1|0|2|MM2S": 5, "1|2|0|CORE": 35}])
    g = ground_edge(dirs, "1|2|0|CORE", "1|0|2|MM2S")
    assert g.reason == GAP_CROSS_DOMAIN
    assert g.accounted is True


def test_async_cdc_gap_is_accounted(tmp_path):
    dirs = _runs(tmp_path, [
        {"1|0|2|DMA_MM2S_0_FINISHED_TASK": 0, "1|0|2|DMA_S2MM_0_FINISHED_TASK": 0},
        {"1|0|2|DMA_MM2S_0_FINISHED_TASK": 9, "1|0|2|DMA_S2MM_0_FINISHED_TASK": 9}])
    g = ground_edge(dirs, "1|0|2|DMA_S2MM_0_FINISHED_TASK",
                    "1|0|2|DMA_MM2S_0_FINISHED_TASK")
    assert g.reason == GAP_ASYNC_CDC
    assert g.accounted is True


def test_async_cdc_classifies_shim_dma_finished_only():
    assert is_async_cdc("1|0|2|DMA_S2MM_0_FINISHED_TASK") is True
    assert is_async_cdc("1|0|2|DMA_MM2S_0_FINISHED_TASK") is True
    assert is_async_cdc("1|0|2|DMA_S2MM_0_START_TASK") is False   # start, not egress
    assert is_async_cdc("1|2|1|DMA_S2MM_0_FINISHED_TASK") is False  # memtile/core, not shim
    assert is_async_cdc("1|2|0|INSTR_VECTOR") is False


def test_ground_edge_cross_domain_exact_carries_reproduction_offset(tmp_path):
    # exact raw offset (30 every run), cross-domain (shim pkt 2 vs core pkt 0):
    # a Gap, but annotated with the reproduction target.
    dirs = _runs(tmp_path, [{"1|0|2|MM2S": 0, "1|2|0|CORE": 30},
                            {"1|0|2|MM2S": 5, "1|2|0|CORE": 35}])
    g = ground_edge(dirs, "1|2|0|CORE", "1|0|2|MM2S")
    assert g == Gap(parent="1|0|2|MM2S", child="1|2|0|CORE",
                    reason=GAP_CROSS_DOMAIN, reproduction_offset=30)


def test_ground_edge_cross_domain_nonexact_no_reproduction_offset(tmp_path):
    dirs = _runs(tmp_path, [{"1|0|2|MM2S": 0, "1|2|0|CORE": 30},
                            {"1|0|2|MM2S": 5, "1|2|0|CORE": 36}])  # range 1
    g = ground_edge(dirs, "1|2|0|CORE", "1|0|2|MM2S")
    assert g == Gap(parent="1|0|2|MM2S", child="1|2|0|CORE",
                    reason=GAP_CROSS_DOMAIN, reproduction_offset=None)


def test_ground_edge_async_cdc_same_domain_is_gap_not_segment(tmp_path):
    # Both shim NoC-egress completions, SAME domain (1|0|2), exact offset 0 ->
    # WOULD be a spurious Segment(0); the async-CDC guard makes it a Gap.
    dirs = _runs(tmp_path, [
        {"1|0|2|DMA_MM2S_0_FINISHED_TASK": 0, "1|0|2|DMA_S2MM_0_FINISHED_TASK": 0},
        {"1|0|2|DMA_MM2S_0_FINISHED_TASK": 9, "1|0|2|DMA_S2MM_0_FINISHED_TASK": 9}])
    g = ground_edge(dirs, "1|0|2|DMA_S2MM_0_FINISHED_TASK",
                    "1|0|2|DMA_MM2S_0_FINISHED_TASK")
    assert g == Gap(parent="1|0|2|DMA_MM2S_0_FINISHED_TASK",
                    child="1|0|2|DMA_S2MM_0_FINISHED_TASK", reason=GAP_ASYNC_CDC)
