"""The runtime-sequence probe: case plan, host layout, safety rules, MLIR text, instruction
decoding and the two tables (push spacing, sync latency)."""
import struct

import pytest

import gen_runtime_seq_probe as g


def case(name):
    return g.plan()[name]


# -- plan ---------------------------------------------------------------------------------------

def test_plan_size_and_families():
    plan = g.plan()
    fam = {f: [c for c in plan.values() if c.family == f] for f in "PW"}
    assert 24 <= len(plan) <= 34
    assert len(fam["P"]) >= 14 and len(fam["W"]) >= 10
    assert len(plan) == len({c.name for c in plan.values()})


def test_plan_covers_the_requested_variables():
    names = set(g.plan())
    # N = 2, 4, 8 in the grouped orders, interleaved, and the S2MM-first order
    for n in (2, 4, 8):
        assert {f"p{n}_mm", f"p{n}_ss"} <= names
    assert {"p4_il", "p8_il"} <= names
    # filler writes before each push: k = 4 and 16 (0 is the plain case)
    assert {"p4_mm_k4", "p4_mm_k16", "p4_ss_k4", "p4_ss_k16"} <= names
    # the same channel twice in a row, against the same pushes spread over channels
    assert {"p8_mm", "p8_mmpair", "p4_same_mm"} <= names
    # sync latency: 1, 2, 4 pairs at 300 / 3000 / 30000 words, and a late sync
    for pairs in (1, 2, 4):
        for words in (300, 3000, 30000):
            assert f"w{pairs}_L{words}" in names
    assert any(n.startswith("w1_late") for n in names)


def test_filler_count_is_the_independent_variable():
    c = case("p4_mm_k16")
    fills = [s for s in c.steps if isinstance(s, g.Fill)]
    pushes = g.measured_pushes(c)
    assert len(pushes) == 4 and all(f.count == 16 for f in fills[:4])
    assert not [s for s in case("p4_mm").steps[:8] if isinstance(s, g.Fill)]


def test_push_orders():
    def order(name):
        return [s.ch for s in g.measured_pushes(case(name))]
    assert order("p2_mm") == ["M0", "S0"] and order("p2_ss") == ["S0", "M0"]
    assert order("p4_mm") == ["M0", "M1", "S0", "S1"]
    assert order("p4_il") == ["M0", "S0", "M1", "S1"]
    assert order("p8_mm") == ["M0", "M1", "M0", "M1", "S0", "S1", "S0", "S1"]
    assert order("p8_mmpair") == ["M0", "M0", "M1", "M1", "S0", "S0", "S1", "S1"]
    assert order("p4_same_ss") == ["S0", "S0", "M0", "M0"]


# -- safety: every task finishes, nothing waits on something that cannot happen ------------------

def test_every_case_validates():
    for c in g.plan().values():
        g.validate(c)


def test_validate_rejects_a_sink_that_is_never_fed():
    bad = g.Case("bad", "P", (g.Push("S0", 8, True), g.Sync("S0")))
    with pytest.raises(ValueError, match="never fed|unfed|words"):
        g.validate(bad)


def test_validate_rejects_a_sync_without_a_token():
    bad = g.Case("bad", "P", (g.Push("M0", 8, False), g.Push("S0", 8, False), g.Sync("S0")))
    with pytest.raises(ValueError, match="token"):
        g.validate(bad)


def test_validate_rejects_a_sync_before_its_data_can_arrive():
    bad = g.Case("bad", "P", (g.Push("S0", 8, True), g.Sync("S0"), g.Push("M0", 8, False)))
    with pytest.raises(ValueError, match="fed|arrive|words"):
        g.validate(bad)


def test_validate_requires_the_last_task_of_a_sink_to_be_awaited():
    bad = g.Case("bad", "P", (g.Push("M0", 8, False), g.Push("S0", 8, True)))
    with pytest.raises(ValueError, match="await|sync"):
        g.validate(bad)


def test_marker_sink_takes_every_marker():
    c = case("w4_L300")
    s1 = [s for s in g.measured_pushes(c) if s.ch == "S1"]
    m1 = [s for s in g.measured_pushes(c) if s.ch == "M1"]
    assert len(s1) == 1 and s1[0].words == sum(s.words for s in m1) == 4 * g.MARKER_WORDS


def test_late_sync_waits_behind_idle_writes():
    c = case("w1_late_L300")
    i = next(i for i, s in enumerate(c.steps) if isinstance(s, g.Sync))
    assert isinstance(c.steps[i - 1], g.Fill) and c.steps[i - 1].count == g.LATE_FILL == 64


# -- host buffer layout ----------------------------------------------------------------------------

def test_buffer_is_sized_from_offset_plus_length():
    for c in g.plan().values():
        lay = g.layout(c)
        top = max(off + s.words for s, off in lay)
        assert g.buffer_words(c) == top
        assert g.buffer_words(c) >= sum(s.words for s in c.steps if isinstance(s, g.Push) and s.ch[0] == "S")


def test_sources_and_destinations_do_not_overlap():
    c = case("p8_mm")
    lay = [(p, o) for p, o in g.layout(c) if not p.tail]
    src = [(o, o + s.words) for s, o in lay if s.ch[0] == "M"]
    dst = [(o, o + s.words) for s, o in lay if s.ch[0] == "S"]
    assert max(e for _, e in src) <= min(b for b, _ in dst)
    assert len(set(src)) == len(src) and len(set(dst)) == len(dst)


# -- MLIR text --------------------------------------------------------------------------------------

def test_mlir_has_one_memcpy_per_push_one_wait_per_sync_and_the_fillers():
    c = case("p4_mm_k4")
    text = g.design_mlir(c)
    assert text.count("aiex.npu.dma_memcpy_nd") == len(g.pushes(c)) == 4 + 4 * g.TAIL_ROUNDS
    assert text.count("aiex.npu.dma_wait") == sum(isinstance(s, g.Sync) for s in c.steps)
    assert text.count("aiex.npu.write32") >= 16
    assert "aie.flow(%t00, DMA : 0, %t00, DMA : 0)" in text
    assert "aie.flow(%t00, DMA : 1, %t00, DMA : 1)" in text
    assert f"memref<{g.buffer_words(c)}xi32>" in text


def test_mlir_traces_start_and_finish_of_all_four_channels():
    text = g.design_mlir(case("p4_mm"))
    for ch in ("MM2S_0", "MM2S_1", "S2MM_0", "S2MM_1"):
        assert f'"DMA_{ch}_START_TASK"' in text and f'"DMA_{ch}_FINISHED_TASK"' in text
    assert text.count("aie.trace.event") == 8   # the carrier shim has no trace unit of its own


def test_large_transfers_stay_within_the_bd_dimension_limit():
    text = g.design_mlir(case("w1_L30000"))
    assert "[%c1, %c1, %c30, %c1000]" in text.replace("%c_", "%c")


# -- instruction stream -------------------------------------------------------------------------------

def blob(ops: list[bytes]) -> bytes:
    body = b"".join(ops)
    return struct.pack("<4I", 0x06030100, 0x0102, len(ops), 16 + len(body)) + body


def w32(addr, val=0):
    return struct.pack("<2I", g.TXN_WRITE, 0) + struct.pack("<IIII", addr, 0, val, 24)[:16]


def bw(addr, words):
    payload = b"".join(struct.pack("<I", w) for w in words)
    return struct.pack("<2I", g.TXN_BLOCKWRITE, 0) + struct.pack("<II", addr, 16 + len(payload)) + payload


def patch():
    return struct.pack("<2I", g.TXN_PATCH, 48) + bytes(40)


def sync():
    return struct.pack("<2I", g.TXN_SYNC, 16) + bytes(8)


def test_decode_ops_walks_to_the_stated_length():
    ops = g.decode_ops(blob([w32(0x1D204), bw(0x1D000, [8] * 8), patch(), sync()]))
    assert [o.kind for o in ops] == ["write32", "blockwrite", "patch", "sync"]
    assert ops[0].offset == 0x1D204 and ops[1].offset == 0x1D000 and ops[1].words == 8


def test_decode_ops_rejects_a_truncated_stream():
    with pytest.raises(ValueError):
        g.decode_ops(blob([w32(1)])[:-4])


def test_pushes_are_the_task_queue_writes_of_the_data_shim():
    q = {"M0": 0x1D214, "M1": 0x1D21C, "S0": 0x1D204, "S1": 0x1D20C}
    stream = blob([w32(0x1D204 | (1 << 25)),               # the carrier column's queue: not a push
                   bw(0x1D000, [8] * 8), patch(), w32(q["M0"]),
                   w32(0x340F0), w32(0x340F0), bw(0x1D020, [8] * 8), patch(), w32(q["S1"]), sync()])
    ops = g.decode_ops(stream)
    pushes = g.push_ops(ops, data_col=0)
    assert [p.channel for p in pushes] == ["M0", "S1"]
    gaps = g.between_pushes(ops, pushes)
    assert gaps == [{"blockwrite": 1, "patch": 1, "write32": 2}]


def test_bd_words_and_data_column_are_found_in_a_real_build_layout():
    assert g.QUEUE_OFFSETS == {0x1D204: "S0", 0x1D20C: "S1", 0x1D214: "M0", 0x1D21C: "M1"}


# -- tables ----------------------------------------------------------------------------------------------

def ev(name, ts):
    return {"name": name, "ts": ts, "pkt_type": 2, "col": 1, "row": 0}


def test_push_rows_pair_the_nth_push_of_a_channel_with_its_nth_start():
    c = case("p8_mmpair")
    starts = {"M0": [100, 1000], "M1": [2000, 3000], "S0": [4100, 5100], "S1": [6100, 7100]}
    events = [ev(f"DMA_{'MM2S' if ch[0] == 'M' else 'S2MM'}_{ch[1]}_START_TASK", t)
              for ch, ts in starts.items() for t in ts]
    rows = g.push_rows(c, events, trace_start=0)
    assert [r["ch"] for r in rows] == ["M0", "M0", "M1", "M1", "S0", "S0", "S1", "S1"]
    assert [r["start"] for r in rows] == [100, 1000, 2000, 3000, 4100, 5100, 6100, 7100]
    assert [r["gap"] for r in rows] == [None, 900, 1000, 1000, 1100, 1000, 1000, 1000]


def test_push_rows_are_relative_to_the_trace_start():
    c = case("p2_mm")
    events = [ev("DMA_MM2S_0_START_TASK", 5100), ev("DMA_S2MM_0_START_TASK", 6000)]
    rows = g.push_rows(c, events, trace_start=5000)
    assert [r["start"] for r in rows] == [100, 1000]


def test_sync_rows_give_finished_to_marker_start():
    c = case("w2_L300")
    events = [ev("DMA_S2MM_0_START_TASK", 100), ev("DMA_S2MM_0_FINISHED_TASK", 500),
              ev("DMA_MM2S_1_START_TASK", 4500),
              ev("DMA_S2MM_0_START_TASK", 4600), ev("DMA_S2MM_0_FINISHED_TASK", 5000),
              ev("DMA_MM2S_1_START_TASK", 9000)]
    rows = g.sync_rows(c, events, trace_start=0)
    assert [r["latency"] for r in rows] == [4000, 4000]
    assert [r["transfer"] for r in rows] == [400, 400]


def test_sync_rows_for_the_no_sync_control_report_the_same_quantity():
    c = case("w1_nosync_L300")
    events = [ev("DMA_S2MM_0_START_TASK", 100), ev("DMA_S2MM_0_FINISHED_TASK", 500),
              ev("DMA_MM2S_1_START_TASK", 300)]
    assert g.sync_rows(c, events, trace_start=0)[0]["latency"] == -200


def test_finish_check_needs_every_start_and_finish_and_a_tail():
    c = case("p2_mm")
    full = [ev("DMA_MM2S_0_START_TASK", 10), ev("DMA_MM2S_0_FINISHED_TASK", 40),
            ev("DMA_S2MM_0_START_TASK", 20), ev("DMA_S2MM_0_FINISHED_TASK", 50)]
    assert g.finish_check(c, full, trace_stop=9000)[0] == "finish ok"
    assert "missing" in g.finish_check(c, full[:-1], trace_stop=9000)[0]
    assert "tail" in g.finish_check(c, full, trace_stop=30)[0]


# -- closing tail: every channel awaited, measured section unchanged ------------------------------------

def test_every_channel_ends_on_an_awaited_token_task_before_the_trailing_writes():
    for c in g.plan().values():
        steps = c.steps
        assert isinstance(steps[-1], g.Fill) and steps[-1].count == g.TAIL_FILL
        last_sync = max(i for i, s in enumerate(steps) if isinstance(s, g.Sync))
        assert not any(isinstance(s, g.Push) for s in steps[last_sync:])
        assert {s.ch for s in steps[last_sync - 3:last_sync + 1]} == set(g.EVENT_ORDER)
        for ch in g.EVENT_ORDER:
            last = max(i for i, s in enumerate(steps) if isinstance(s, g.Push) and s.ch == ch)
            assert steps[last].token and steps[last].tail
            assert any(isinstance(s, g.Sync) and s.ch == ch for s in steps[last:])


def test_validate_rejects_an_unawaited_mm2s_task():
    bad = g.Case("bad", "P", (g.Push("M0", 8, False), g.Push("S0", 8, True), g.Sync("S0"), *g.closing_tail()[:-4]))
    with pytest.raises(ValueError, match="await|token"):
        g.validate(bad)


def test_validate_rejects_a_case_without_the_closing_tail():
    bad = g.Case("bad", "P", (g.Push("M0", 8, True), g.Push("S0", 8, True), g.Sync("M0"), g.Sync("S0")))
    with pytest.raises(ValueError, match="tail"):
        g.validate(bad)


def legacy_layout(c):
    """Set 9's layout of the measured pushes: sources first, destinations after."""
    m = g.measured_pushes(c)
    nxt = {"M": 0, "S": sum(p.words for p in m if p.ch[0] == "M")}
    out = []
    for p in m:
        out.append(nxt[p.ch[0]])
        nxt[p.ch[0]] += p.words
    return out


def test_measured_offsets_are_the_ones_set_9_ran_with_and_the_tail_lies_above():
    for c in g.plan().values():
        lay = g.layout(c)
        measured = [o for p, o in lay if not p.tail]
        assert measured == legacy_layout(c)
        top = max(o + p.words for p, o in lay if not p.tail)
        tail = sorted((o, o + p.words) for p, o in lay if p.tail)
        assert tail[0][0] >= top
        assert all(a[1] <= b[0] for a, b in zip(tail, tail[1:]))


def test_tail_reuses_the_low_bd_ids_and_measured_ids_are_unchanged():
    import re
    c = case("w4_L300")
    ids = [int(x) for x in re.findall(r"id = (\d+) : i64", g.design_mlir(c))]
    n = len(g.measured_pushes(c))
    assert ids[:n] == list(range(n))
    assert ids[n:] == list(g.TAIL_BD_IDS) * g.TAIL_ROUNDS
    assert max(ids) < 16


def test_finish_check_accepts_a_tail_cut_short_but_not_a_measured_event_missing():
    c = case("p2_mm")
    measured = [ev("DMA_MM2S_0_START_TASK", 10), ev("DMA_MM2S_0_FINISHED_TASK", 40),
                ev("DMA_S2MM_0_START_TASK", 20), ev("DMA_S2MM_0_FINISHED_TASK", 50)]
    tail_part = [ev("DMA_MM2S_0_START_TASK", 900), ev("DMA_MM2S_1_START_TASK", 1900)]
    assert g.finish_check(c, measured + tail_part, trace_stop=None)[0] == "finish ok"
    assert "missing" in g.finish_check(c, measured[1:] + tail_part, trace_stop=None)[0]
    assert "tail" in g.finish_check(c, measured, trace_stop=None)[0]
    no_fin = [e for e in measured if e["name"] != "DMA_S2MM_0_FINISHED_TASK"]
    tail_fin = tail_part + [ev("DMA_S2MM_0_START_TASK", 1000), ev("DMA_S2MM_0_FINISHED_TASK", 1100)]
    assert "missing S0 FINISHED" in g.finish_check(c, no_fin + tail_fin, trace_stop=None)[0]


def test_sync_rows_ignore_the_tail_tasks_of_s0():
    c = case("w1_L300")
    events = [ev("DMA_S2MM_0_START_TASK", 100), ev("DMA_S2MM_0_FINISHED_TASK", 500),
              ev("DMA_MM2S_1_START_TASK", 4500),
              ev("DMA_S2MM_0_START_TASK", 9000), ev("DMA_S2MM_0_FINISHED_TASK", 9100)]
    rows = g.sync_rows(c, events, trace_start=0)
    assert len(rows) == 1 and rows[0]["latency"] == 4000


def test_case_table_lists_only_the_measured_pushes(tmp_path):
    import json
    c = case("p2_mm")
    d = tmp_path / c.name
    (d / "r").mkdir(parents=True)
    (d / "case.json").write_text(json.dumps({"between": [{} for _ in g.pushes(c)]}))
    events = [ev("DMA_MM2S_0_START_TASK", 10), ev("DMA_S2MM_0_START_TASK", 20)] + \
             [ev("DMA_MM2S_0_START_TASK", 100 * i) for i in range(1, 4)]
    real_load = g.load_run
    g.load_run = lambda case_dir, run: (events, 0, None)
    try:
        text = g.case_table(c, d, ["r"])
    finally:
        g.load_run = real_load
    rows = [l for l in text.splitlines() if l.strip().startswith(("0  ", "1  ", "2  "))]
    assert len(rows) == 2 and "2 pushes + 12 tail" in text


def test_tokens_and_syncs_are_level_on_every_channel():
    for c in g.plan().values():
        for ch in g.EVENT_ORDER:
            tok = sum(1 for s in c.steps if isinstance(s, g.Push) and s.ch == ch and s.token)
            syn = sum(1 for s in c.steps if isinstance(s, g.Sync) and s.ch == ch)
            assert tok == syn, (c.name, ch)


def test_all_token_case_drains_its_surplus_before_the_tail():
    c = case("p8_mm_alltok")
    first_tail = next(i for i, s in enumerate(c.steps) if isinstance(s, g.Push) and s.tail)
    before = c.steps[:first_tail]
    for ch in g.EVENT_ORDER:
        tok = sum(1 for s in before if isinstance(s, g.Push) and s.ch == ch and s.token)
        syn = sum(1 for s in before if isinstance(s, g.Sync) and s.ch == ch)
        assert tok == syn == 2, ch
    assert g.drain_syncs(list(case("p8_mm").steps[:12])) == []


def test_validate_rejects_a_surplus_token():
    bad = g.Case("bad", "P", (g.Push("M0", 8, True), g.Push("S0", 8, True), g.Push("M0", 8, True),
                              g.Push("S0", 8, True), g.Sync("S0"), g.Sync("S0"), g.Sync("M0"), *g.closing_tail()))
    with pytest.raises(ValueError, match="surplus|tokens"):
        g.validate(bad)


def test_decode_ops_reads_a_maskpoll():
    mp = struct.pack("<2I", g.TXN_MASKPOLL, 0) + struct.pack("<IIIII", 0x1D228, 0, 0, 0x400000, 28)
    ops = g.decode_ops(blob([w32(0x1D204), mp, sync()]))
    assert [o.kind for o in ops] == ["write32", "maskpoll", "sync"] and ops[1].offset == 0x1D228


def test_a_build_stops_when_the_runner_sizes_the_buffer_differently(monkeypatch, tmp_path):
    case = g.plan()["w2_L300"]
    need = 4 * g.buffer_words(case)

    class Out:
        def __init__(self, rc, out, err=""):
            self.returncode, self.stdout, self.stderr = rc, out, err

    monkeypatch.setattr(g.subprocess, "run", lambda *a, **k: Out(0, f"0 {need}\n1 65536\n"))
    g.check_runner_sizing(case, tmp_path / "insts.bin", tmp_path / "runner")
    monkeypatch.setattr(g.subprocess, "run", lambda *a, **k: Out(0, f"0 {need - 1200}\n1 65536\n"))
    with pytest.raises(SystemExit, match="layout needs"):
        g.check_runner_sizing(case, tmp_path / "insts.bin", tmp_path / "runner")
    monkeypatch.setattr(g.subprocess, "run", lambda *a, **k: Out(2, "", "opcode 0x02 is not decoded"))
    with pytest.raises(SystemExit, match="cannot size"):
        g.check_runner_sizing(case, tmp_path / "insts.bin", tmp_path / "runner")
