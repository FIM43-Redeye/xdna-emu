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


# -- probe 9d: bracketed syncs, fill sweep, dummy pair, no-sync variants -------------------------------------

def case9(name):
    return g.plan_9d()[name]


MARKED = ("w1_late_L300", "w1_L300", "w1_L3000", "w1_L30000", "w2_L30000")


def test_plan_9d_names_are_unique_validate_and_leave_the_base_plan_alone():
    p9 = g.plan_9d()
    assert len(p9) == 25
    for c in p9.values():
        g.validate(c)
    base = g.plan()
    assert len(base) == 30          # probe 9c's plan is unchanged
    for name in ("w1_late_L300", "w1_L300", "w1_L3000", "w1_late_L3000"):   # shared cases are the same plan
        assert p9[name].steps == base[name].steps and p9[name].events == g.DEFAULT_EVENTS


def test_plan_9d_has_every_requested_case():
    names = set(g.plan_9d())
    for base in MARKED:
        assert {f"{base}_br", f"{base}_tw"} <= names
    assert {"w1_L300", "w1_L3000", "w1_L1000", "w1_L10000", "w1_L20000", "w1_late_L30000"} <= names
    assert {"w1_f16_L3000", "w1_f32_L3000", "w1_late_L3000", "w1_f128_L3000"} <= names
    assert {"w1_latenosync_L300", "w1_latenosync_L3000", "w2_dummy_L3000", "w2_nosync_L300"} <= names


def test_default_events_are_the_eight_channel_events_of_probe_9c():
    assert len(g.DEFAULT_EVENTS) == 8
    assert g.DEFAULT_EVENTS[0] == "DMA_MM2S_0_START_TASK" and g.DEFAULT_EVENTS[-1] == "DMA_S2MM_1_FINISHED_TASK"
    assert case("w2_L300").events == g.DEFAULT_EVENTS


def test_bracket_events_trade_the_two_mm2s_finishes_for_the_two_user_events():
    ev9 = g.BRACKET_EVENTS
    assert len(ev9) == 8 and len(set(ev9)) == 8
    assert set(g.DEFAULT_EVENTS) - set(ev9) == {"DMA_MM2S_0_FINISHED_TASK", "DMA_MM2S_1_FINISHED_TASK"}
    assert set(ev9) - set(g.DEFAULT_EVENTS) == {"USER_EVENT_0", "USER_EVENT_1"}


def test_marks_surround_every_measured_sync_and_nothing_else():
    for base in MARKED:
        c = case9(f"{base}_br")
        steps = c.steps
        assert c.events == g.BRACKET_EVENTS
        n_meas_syncs = 0
        for i, s in enumerate(steps):
            if isinstance(s, g.Mark):
                nxt, prv = steps[i + 1], steps[i - 1]
                assert (s.which == "A" and isinstance(nxt, g.Sync)) or (s.which == "B" and isinstance(prv, g.Sync))
        first_tail = next(i for i, s in enumerate(steps) if isinstance(s, g.Push) and s.tail)
        assert not any(isinstance(s, g.Mark) for s in steps[first_tail:])
        measured_syncs = [i for i, s in enumerate(steps[:first_tail]) if isinstance(s, g.Sync)]
        assert measured_syncs and all(isinstance(steps[i - 1], g.Mark) and steps[i - 1].which == "A" and
                                      isinstance(steps[i + 1], g.Mark) and steps[i + 1].which == "B"
                                      for i in measured_syncs)


def test_a_twin_is_its_bracketed_case_without_the_marks_and_traces_the_same_events():
    for base in MARKED:
        br, tw = case9(f"{base}_br"), case9(f"{base}_tw")
        assert tw.events == br.events
        assert tw.steps == tuple(s for s in br.steps if not isinstance(s, g.Mark))
        assert not any(isinstance(s, g.Mark) for s in tw.steps)


def test_bracketed_streams_are_shaped_like_their_unbracketed_cases():
    """The measured section (everything before the closing tail) of a twin is its unbracketed case's."""
    def measured(c):
        k = next(i for i, s in enumerate(c.steps) if isinstance(s, g.Push) and s.tail)
        return c.steps[:k]
    for base in MARKED:
        plain = case9(base) if base in g.plan_9d() else g.plan()[base]
        assert measured(case9(f"{base}_tw")) == measured(plain)


def test_marks_are_event_generate_writes_to_the_traced_shim_not_the_carrier():
    text = g.design_mlir(case9("w1_L300_br"))
    n_marks = sum(isinstance(s, g.Mark) for s in case9("w1_L300_br").steps)
    assert n_marks == 4
    writes = [l for l in text.splitlines() if "aiex.npu.write32" in l and "column = 0 : i32" in l]
    assert len(writes) == n_marks
    assert f"%c_evgen = arith.constant {g.EVENT_GENERATE} : i32" in text
    assert "%c_mark_A = arith.constant 126 : i32" in text and "%c_mark_B = arith.constant 127 : i32" in text
    assert g.EVENT_GENERATE == 0x34008 and g.MARK_EVENTS == {"A": ("USER_EVENT_0", 126), "B": ("USER_EVENT_1", 127)}
    assert text.count('aie.trace.event <"USER_EVENT_0">') == 1 and text.count('aie.trace.event <"USER_EVENT_1">') == 1
    assert "DMA_MM2S_0_FINISHED_TASK" not in text and "DMA_S2MM_0_FINISHED_TASK" in text
    twin = g.design_mlir(case9("w1_L300_tw"))
    assert "column = 0 : i32" not in twin and 'aie.trace.event <"USER_EVENT_0">' in twin


def test_unmarked_default_cases_keep_the_nine_c_trace_config():
    text = g.design_mlir(case("w1_L300"))
    assert text.count("aie.trace.event") == 8 and "USER_EVENT" not in text


def test_validate_rejects_a_mark_that_does_not_touch_a_sync():
    base = case9("w1_L300_br").steps
    i = next(i for i, s in enumerate(base) if isinstance(s, g.Mark))
    bad = g.Case("bad", "W", base[:i + 1] + (g.Fill(3),) + base[i + 1:], g.BRACKET_EVENTS)
    with pytest.raises(ValueError, match="mark"):
        g.validate(bad)


def test_validate_rejects_marks_without_their_trace_events():
    c = case9("w1_L300_br")
    with pytest.raises(ValueError, match="USER_EVENT"):
        g.validate(g.Case(c.name, c.family, c.steps, g.DEFAULT_EVENTS))


def test_fill_sweep_is_the_independent_variable_before_the_sync():
    for name, n in (("w1_L3000", 0), ("w1_f16_L3000", 16), ("w1_f32_L3000", 32),
                    ("w1_late_L3000", 64), ("w1_f128_L3000", 128)):
        c = case9(name)
        i = next(i for i, s in enumerate(c.steps) if isinstance(s, g.Sync))
        if n:
            assert isinstance(c.steps[i - 1], g.Fill) and c.steps[i - 1].count == n
        else:
            assert isinstance(c.steps[i - 1], g.Push)
        assert [s.words for s in g.measured_pushes(c) if s.ch == "S0"] == [3000]


def test_late_flag_still_means_the_sixty_four_fillers():
    assert g._w_case("x", 1, 300, late=True).steps == g._w_case("x", 1, 300, fill=g.LATE_FILL).steps


def test_transfer_length_sweep():
    for name, words in (("w1_L1000", 1000), ("w1_L10000", 10000), ("w1_L20000", 20000), ("w1_late_L30000", 30000)):
        assert [s.words for s in g.measured_pushes(case9(name)) if s.ch == "S0"] == [words]
    assert any(isinstance(s, g.Fill) and s.count == g.LATE_FILL for s in case9("w1_late_L30000").steps)


def test_late_nosync_pushes_the_marker_after_the_fillers_with_no_sync_before_it():
    for name in ("w1_latenosync_L300", "w1_latenosync_L3000"):
        c = case9(name)
        kinds = [type(s).__name__ + (":" + s.ch if hasattr(s, "ch") else "") for s in c.steps]
        i_m1 = kinds.index("Push:M1")
        assert kinds[i_m1 - 2:i_m1] == ["Push:S0", "Fill"]
        assert c.steps[i_m1 - 1].count == g.LATE_FILL
        assert "Sync:S0" not in kinds[:i_m1] and kinds[i_m1 + 1:i_m1 + 3] == ["Sync:S0", "Sync:S1"]


def test_two_pair_no_sync_case_pushes_both_pairs_before_any_sync():
    c = case9("w2_nosync_L300")
    first_sync = next(i for i, s in enumerate(c.steps) if isinstance(s, g.Sync))
    head = [s.ch for s in c.steps[:first_sync] if isinstance(s, g.Push)]
    assert head == ["S1", "M0", "S0", "M1", "M0", "S0", "M1"]
    after = [s.ch for s in c.steps[first_sync:first_sync + 3]]
    assert after == ["S0", "S0", "S1"]


def test_dummy_pair_is_an_eight_word_task_and_its_sync_before_the_real_pairs():
    c = case9("w2_dummy_L3000")
    head = c.steps[:4]
    assert [type(s).__name__ for s in head] == ["Push", "Push", "Push", "Sync"]
    assert head[0].ch == "S1" and not head[0].dummy
    assert (head[1].ch, head[1].words, head[1].token, head[1].dummy) == ("M0", 8, False, True)
    assert (head[2].ch, head[2].words, head[2].token, head[2].dummy) == ("S0", 8, True, True)
    assert head[3].ch == "S0"
    real_s0 = [s for s in g.measured_pushes(c) if s.ch == "S0" and not s.dummy]
    assert [s.words for s in real_s0] == [3000, 3000]
    assert [s.words for s in g.measured_pushes(c) if s.ch == "S1"] == [2 * g.MARKER_WORDS]


def test_sync_rows_skip_the_dummy_pair():
    c = case9("w2_dummy_L3000")
    events = [ev("DMA_S2MM_0_START_TASK", 50), ev("DMA_S2MM_0_FINISHED_TASK", 80),
              ev("DMA_S2MM_0_START_TASK", 1000), ev("DMA_S2MM_0_FINISHED_TASK", 4000),
              ev("DMA_MM2S_1_START_TASK", 9000),
              ev("DMA_S2MM_0_START_TASK", 9500), ev("DMA_S2MM_0_FINISHED_TASK", 12500),
              ev("DMA_MM2S_1_START_TASK", 17000)]
    rows = g.sync_rows(c, events, trace_start=0)
    assert [r["latency"] for r in rows] == [5000, 4500] and [r["s0_start"] for r in rows] == [1000, 9500]


def uev(n, ts):
    return {"name": f"USER_EVENT_{n}", "ts": ts, "pkt_type": 2, "col": 0, "row": 0}


def test_bracket_rows_split_a_sync_into_issue_finish_resume_next_push():
    c = case9("w1_L3000_br")
    events = [ev("DMA_MM2S_0_START_TASK", 50), ev("DMA_S2MM_0_START_TASK", 100), uev(0, 700), ev("DMA_S2MM_0_FINISHED_TASK", 3300), uev(1, 8000),
              ev("DMA_MM2S_1_START_TASK", 8700),
              uev(0, 9000), ev("DMA_S2MM_1_FINISHED_TASK", 9500), uev(1, 14000), ev("DMA_MM2S_0_START_TASK", 15000)]
    rows = g.bracket_rows(c, events, trace_start=0)
    assert [r["sync"] for r in rows] == ["S0", "S1"]
    r = rows[0]
    assert (r["mark_a"], r["mark_b"], r["fin"], r["next"]) == (700, 8000, 3300, 8700)
    assert (r["a_to_fin"], r["fin_to_b"], r["a_to_b"], r["b_to_next"]) == (2600, 4700, 7300, 700)
    assert rows[1]["next_ch"] == "M0" and rows[1]["b_to_next"] == 1000


def test_bracket_rows_use_the_dummy_aware_task_index():
    c = case9("w2_dummy_L3000")
    assert g.bracket_rows(c, [], trace_start=0) == []      # unmarked case: nothing to bracket


def test_finish_check_of_a_bracketed_case_wants_marks_not_the_mm2s_finishes():
    c = case9("w1_L300_br")
    full = [ev("DMA_MM2S_0_START_TASK", 5), ev("DMA_MM2S_1_START_TASK", 60), ev("DMA_S2MM_0_START_TASK", 10),
            ev("DMA_S2MM_0_FINISHED_TASK", 50), ev("DMA_S2MM_1_START_TASK", 1), ev("DMA_S2MM_1_FINISHED_TASK", 90),
            uev(0, 20), uev(1, 55), uev(0, 70), uev(1, 80)]
    tail = [ev("DMA_MM2S_0_START_TASK", 900)]
    assert g.finish_check(c, full + tail, trace_stop=None)[0] == "finish ok"
    assert "USER_EVENT_1" in g.finish_check(c, [e for e in full + tail if e != full[-1]], trace_stop=None)[0]


def test_marks_in_ops_counts_event_generate_writes_at_the_traced_shim():
    ops = [g.Op("write32", 0, 0, 0x34008), g.Op("write32", 1, 0, 0x34008), g.Op("write32", 0, 0, 0x340F0),
           g.Op("write32", 0, 0, 0x34008)]
    assert g.marks_in_ops(ops) == 2


def traced_tail_events(c):
    """Events the closing tail puts in the trace: a traced START or FINISHED per tail task."""
    n = 0
    for p in g.pushes(c):
        if p.tail:
            d, num, _ = g.CHANNELS[p.ch]
            n += sum(1 for w in ("START", "FINISHED") if f"DMA_{d}_{num}_{w}_TASK" in c.events)
    return n


def test_a_bracketed_case_has_the_tail_volume_of_probe_9c_cases():
    """With two slots given to user events a three-round tail traces 18 events, not 24: the extra round keeps the
    number of packets that follow the deciding events as large as in probe 9c."""
    ref = traced_tail_events(case("w1_L300"))
    assert ref == 24
    for base in MARKED:
        for suffix in ("br", "tw"):
            c = case9(f"{base}_{suffix}")
            assert traced_tail_events(c) == ref
            assert sum(1 for p in g.pushes(c) if p.tail) == 16


def test_tail_rounds_scale_the_tail_and_its_bd_ids():
    import re
    c = case9("w1_L300_br")
    ids = [int(x) for x in re.findall(r"id = (\d+) : i64", g.design_mlir(c))]
    n = len(g.measured_pushes(c))
    assert ids[n:] == list(g.TAIL_BD_IDS) * 4 and max(ids) < 16
    assert [s for s in c.steps if isinstance(s, g.Push) and s.tail][:4] == \
        [g.Push(ch, g.TAIL_WORDS, True, True) for ch in g.TAIL_ORDER]
    assert len(g.closing_tail(4)) == len(g.closing_tail(2)) * 2 and g.closing_tail() == g.closing_tail(g.TAIL_ROUNDS)


# -- probe 9e: dense transfer-length sweep with marks, fill and dummy variants ---------------------------------

SWEEP_9E = (600, 1000, 1500, 2000, 2500, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 15000, 20000)
DUMMY_9E = (1000, 3000, 8000, 20000)
TWINS_9E = (2000, 8000)


def case9e(name):
    return g.plan_9e()[name]


def test_plan_9e_is_small_unique_valid_and_leaves_the_other_plans_alone():
    p = g.plan_9e()
    assert 20 <= len(p) <= 30
    for c in p.values():
        g.validate(c)
    assert len(g.plan()) == 30 and len(g.plan_9d()) == 25


def test_plan_9e_starts_with_the_positive_control_which_is_probe_9ds_bracketed_l3000():
    p = g.plan_9e()
    first = next(iter(p.values()))
    assert first.name == "w1_L3000_br"
    ref = g.plan_9d()["w1_L3000_br"]
    assert first.steps == ref.steps and first.events == ref.events


def test_plan_9e_has_every_requested_case():
    names = set(g.plan_9e())
    assert {f"w1_L{n}_br" for n in SWEEP_9E} <= names
    assert {"w1_f16_L3000_br", "w1_f32_L3000_br"} <= names
    assert {f"w2_dummy_L{n}_br" for n in DUMMY_9E} <= names
    assert {f"w1_L{n}_tw" for n in TWINS_9E} == {n for n in names if n.endswith("_tw")}


def test_every_9e_case_is_bracketed_or_a_twin_with_the_bracket_slots_and_four_tail_rounds():
    for c in g.plan_9e().values():
        assert c.events == g.BRACKET_EVENTS
        assert sum(1 for p in g.pushes(c) if p.tail) == 16
        assert any(isinstance(s, g.Mark) for s in c.steps) == c.name.endswith("_br")


def test_9e_sweep_cases_are_one_pair_with_no_filler_and_the_requested_length():
    for n in SWEEP_9E:
        c = case9e(f"w1_L{n}_br")
        meas = g.measured_pushes(c)
        assert [p.words for p in meas if p.ch == "S0"] == [n]
        assert not any(isinstance(s, g.Fill) and s.count != g.TAIL_FILL for s in c.steps)


def test_9e_fill_cases_put_the_fillers_between_the_s0_push_and_the_first_mark():
    for n in (16, 32):
        steps = case9e(f"w1_f{n}_L3000_br").steps
        i = next(k for k, s in enumerate(steps) if isinstance(s, g.Fill) and s.count == n)
        assert isinstance(steps[i - 1], g.Push) and steps[i - 1].ch == "S0" and steps[i - 1].words == 3000
        assert isinstance(steps[i + 1], g.Mark) and steps[i + 1].which == "A"


def test_9e_dummy_cases_bracket_both_real_pairs_but_not_the_dummy_sync():
    for n in DUMMY_9E:
        c = case9e(f"w2_dummy_L{n}_br")
        assert [p.words for p in g.measured_pushes(c) if p.ch == "S0" and not p.dummy] == [n, n]
        syncs = [i for i, s in enumerate(c.steps) if isinstance(s, g.Sync)]
        first_tail = next(i for i, s in enumerate(c.steps) if isinstance(s, g.Push) and s.tail)
        meas = [i for i in syncs if i < first_tail]
        assert not isinstance(c.steps[meas[0] - 1], g.Mark)           # the dummy's sync
        assert all(isinstance(c.steps[i - 1], g.Mark) for i in meas[1:])
        assert len(meas) == 4        # dummy, two pairs, final S1


def test_9e_twins_are_their_bracketed_cases_without_the_marks():
    for n in TWINS_9E:
        br, tw = case9e(f"w1_L{n}_br"), case9e(f"w1_L{n}_tw")
        assert tw.steps == tuple(s for s in br.steps if not isinstance(s, g.Mark))


def test_9e_run_count_fits_four_calls_of_twenty():
    assert 3 * len(g.plan_9e()) <= 80


# -- probe 9f: sampling-point edges of the wait, and prior commands that end on different waits -----------------

EDGES_9F = (tuple(range(200, 701, 50)) + (800, 900, 1000) + tuple(range(1900, 2151, 50)) + (2500, 2600, 2700, 2800, 2900)
            + tuple(range(6250, 7051, 100)) + tuple(range(8050, 8951, 100)) + (1500, 2000, 3000))
PRIOR_9F = ("k_a_L30000", "k_c_L30000", "k_f_L30000", "k_e_L3000")


def case9f(name):
    return g.plan_9f()[name]


def test_plan_9f_is_unique_valid_and_leaves_the_other_plans_alone():
    p = g.plan_9f()
    for c in p.values():
        g.validate(c)
    assert len(g.plan()) == 30 and len(g.plan_9d()) == 25 and len(g.plan_9e()) == 26


def test_plan_9f_starts_with_the_positive_control_which_is_probe_9es():
    first = next(iter(g.plan_9f().values()))
    ref = g.plan_9e()["w1_L3000_br"]
    assert first.name == "w1_L3000_br" and first.steps == ref.steps and first.events == ref.events


def test_plan_9f_has_every_requested_case_and_no_other():
    names = set(g.plan_9f())
    want = {f"w1_L{n}_br" for n in EDGES_9F} | set(PRIOR_9F) | {"w1_f128_L3000_br"}
    assert names == want


def test_9f_length_cases_that_exist_in_probe_9e_are_the_same_plans():
    p9e = g.plan_9e()
    for n in (1000, 1500, 2000, 2500, 3000):
        assert case9f(f"w1_L{n}_br").steps == p9e[f"w1_L{n}_br"].steps


def test_9f_length_cases_are_one_bracketed_pair_of_the_requested_length():
    for n in EDGES_9F:
        c = case9f(f"w1_L{n}_br")
        assert c.events == g.BRACKET_EVENTS and sum(isinstance(s, g.Mark) for s in c.steps) == 4
        assert [p.words for p in g.measured_pushes(c) if p.ch == "S0"] == [n]


def test_9f_fill_case_is_the_bracketed_length_case_with_128_writes_after_the_s0_push():
    steps = case9f("w1_f128_L3000_br").steps
    i = next(k for k, s in enumerate(steps) if isinstance(s, g.Fill) and s.count == 128)
    assert isinstance(steps[i - 1], g.Push) and steps[i - 1].ch == "S0" and steps[i - 1].words == 3000
    assert isinstance(steps[i + 1], g.Mark) and steps[i + 1].which == "A"
    assert g.Fill(g.TAIL_FILL) == steps[-1]


def test_9f_prior_command_cases_have_no_marks_and_the_bracket_event_slots():
    for n in PRIOR_9F:
        c = case9f(n)
        assert c.events == g.BRACKET_EVENTS and not any(isinstance(s, g.Mark) for s in c.steps)


def test_9f_prior_command_cases_open_with_a_marker_pair_then_two_ordinary_rounds():
    for n in PRIOR_9F:
        s = case9f(n).steps
        assert list(s[:4]) == [g.Push("S1", 8, True), g.Push("M1", 8, True), g.Sync("M1"), g.Sync("S1")]
        assert list(s[4:4 + len(g.closing_tail(2))]) == g.closing_tail(2)


def test_9f_prior_command_last_round_is_all_tail_tasks_and_ends_with_the_trailing_fill():
    for n in PRIOR_9F:
        s = case9f(n).steps
        last = s[4 + len(g.closing_tail(2)):]
        assert [type(x) for x in last] == [g.Push] * 4 + [g.Sync] * 4 + [g.Fill]
        assert all(x.tail and x.token for x in last[:4]) and last[-1] == g.Fill(32)


def test_9f_prior_command_cases_differ_in_which_path_is_long_and_the_order_of_the_last_syncs():
    def shape(n):
        last = case9f(n).steps[4 + len(g.closing_tail(2)):]
        return [(x.ch, x.words) for x in last[:4]], [x.ch for x in last[4:8]]
    long_m0s0 = [("M0", 30000), ("S0", 30000), ("M1", 8), ("S1", 8)]
    assert shape("k_a_L30000") == (long_m0s0, ["M1", "S1", "M0", "S0"])
    assert shape("k_c_L30000") == (long_m0s0, ["M1", "S1", "S0", "M0"])
    assert shape("k_f_L30000") == ([("M0", 8), ("S0", 8), ("M1", 30000), ("S1", 30000)], ["M0", "S0", "M1", "S1"])
    assert shape("k_e_L3000") == ([("M0", 3000), ("S0", 3000), ("M1", 8), ("S1", 8)], ["M1", "S1", "M0", "S0"])


def test_9f_selectable_from_the_command_line():
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert g.main(["list", "--plan", "9f"]) == 0
    assert "k_a_L30000" in buf.getvalue() and "w1_f128_L3000_br" in buf.getvalue()


# -- probe 9g: fine sweeps of the transfer length around three levels -------------------------------------------

PARTS_9G = {"a": tuple(range(8440, 8641, 20)), "b": tuple(range(2600, 2801, 20)), "c": tuple(range(240, 601, 40))}


def test_plan_9g_is_exactly_the_three_fine_sweeps_plus_the_control_and_validates():
    p = g.plan_9g()
    want = {"w1_L3000_br"} | {f"w1_L{n}_br" for ns in PARTS_9G.values() for n in ns}
    assert set(p) == want and len(p) == 11 + 11 + 10 + 1
    assert next(iter(p)) == "w1_L3000_br"
    for c in p.values():
        g.validate(c)


def test_9g_cases_are_the_same_plans_as_the_other_bracketed_one_pair_cases():
    p9g, p9f = g.plan_9g(), g.plan_9f()
    for name in p9g:
        if name in p9f:
            assert p9g[name].steps == p9f[name].steps and p9g[name].events == p9f[name].events
    for n in (240, 8440, 2620):
        c = p9g[f"w1_L{n}_br"]
        assert c.events == g.BRACKET_EVENTS and sum(isinstance(s, g.Mark) for s in c.steps) == 4
        assert [x.words for x in g.measured_pushes(c) if x.ch == "S0"] == [n]


def test_9g_selectable_from_the_command_line_and_leaves_other_plans_alone():
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert g.main(["list", "--plan", "9g"]) == 0
    assert "w1_L8640_br" in buf.getvalue() and "w1_L240_br" in buf.getvalue()
    assert len(g.plan_9f()) == 51


# -- probe 9h: a prior-command case followed by long waits, and a fine sweep just past one level ---------------------

LONG_9H = tuple(range(7300, 8101, 100))
FINE_9H = tuple(range(8500, 8541, 4))
PAIRED_9H = (9500, 11500)


def test_plan_9h_is_the_two_sweeps_the_prior_command_case_and_the_control_and_validates():
    p = g.plan_9h()
    want = {"w1_L3000_br", "k_a_L30000"} | {f"w1_L{n}_br" for n in LONG_9H + FINE_9H + PAIRED_9H}
    assert set(p) == want and len(p) == 2 + 9 + 11 + 2
    assert next(iter(p)) == "w1_L3000_br"
    for c in p.values():
        g.validate(c)


def test_9h_shares_its_plans_with_the_earlier_probes():
    p9h, p9f = g.plan_9h(), g.plan_9f()
    assert p9h["k_a_L30000"].steps == p9f["k_a_L30000"].steps and p9h["w1_L3000_br"].steps == p9f["w1_L3000_br"].steps
    for n in LONG_9H + FINE_9H + PAIRED_9H:
        c = p9h[f"w1_L{n}_br"]
        assert c.events == g.BRACKET_EVENTS and sum(isinstance(s, g.Mark) for s in c.steps) == 4
        assert [x.words for x in g.measured_pushes(c) if x.ch == "S0"] == [n]


def test_9h_selectable_from_the_command_line():
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert g.main(["list", "--plan", "9h"]) == 0
    assert "w1_L8540_br" in buf.getvalue() and "k_a_L30000" in buf.getvalue()
