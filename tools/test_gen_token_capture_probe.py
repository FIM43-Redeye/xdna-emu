"""The task-complete token capture probe: designs, routing text, wait rules, termination, decoding."""
import re

import gen_token_capture_probe as g


def _base():
    return {n: g.designs()[n] for n in g.BASE_DESIGNS}


def test_the_base_designs_cover_every_dma_channel_type_the_probe_measures():
    got = {(p.kind, p.direction, p.index) for d in _base().values() for p in d.probes}
    want = ({("core", "s2mm", i) for i in range(2)} | {("core", "mm2s", i) for i in range(2)}
            | {("mem", "s2mm", i) for i in range(6)} | {("mem", "mm2s", i) for i in range(6)}
            | {("shim", "s2mm", i) for i in range(2)} | {("shim", "mm2s", i) for i in range(2)})
    assert got == want


def test_each_base_design_has_a_half_suppressed_and_a_doubled_variant():
    table = g.designs()
    for n, d in _base().items():
        half, dbl = table[f"{n}-half"], table[f"{n}-dbl"]
        assert half.probes == d.probes and half.suppressed == frozenset(range(1, len(d.probes), 2))
        assert dbl.probes == tuple(q for p in d.probes for q in (p, p)) and not dbl.suppressed
        assert sum(half.expected().values()) == len(d.probes) - len(half.suppressed)
        assert sum(dbl.expected().values()) == 2 * len(d.probes)


def test_token_ids_are_unique_per_channel_and_clear_of_the_reserved_ids():
    reserved = {g.FILLER_ID, g.SHIM_CONTROLLER_ID, g.SHIM1_CONTROLLER_ID, 0}
    for d in g.designs().values():
        by_channel: dict[tuple, int] = {}
        for p in d.probes:
            assert p.token_id not in reserved and 0 < p.token_id <= 31
            assert by_channel.setdefault((p.tile, p.direction, p.index), p.token_id) == p.token_id
        assert len(set(by_channel.values())) == len(by_channel)


def test_each_token_id_has_one_packet_flow_from_the_tile_control_port_to_the_capture_dma_keeping_the_header():
    for d in g.designs().values():
        text = g.design_mlir(d)
        for tid, tile in {(p.token_id, p.tile) for p in d.probes}:
            flow = re.search(rf"aie\.packet_flow\({tid}\) \{{\s*aie\.packet_source<{re.escape(tile)}, TileControl : 0>\s*"
                             rf"aie\.packet_dest<%t00, DMA : 0>\s*\}} \{{keep_pkt_header = true\}}", text)
            assert flow, (d.name, tid)
            assert len(re.findall(rf"aie\.packet_flow\({tid}\)", text)) == 1


def test_the_filler_has_its_own_packet_flow_from_shim_mm2s_0_into_the_capture_channel():
    text = g.design_mlir(g.designs()["cmp"])
    assert re.search(rf"aie\.packet_flow\({g.FILLER_ID}\) \{{\s*aie\.packet_source<%t00, DMA : 0>\s*"
                     r"aie\.packet_dest<%t00, DMA : 0>\s*\} \{keep_pkt_header = true\}", text)
    assert f"packet = <pkt_id = {g.FILLER_ID}, pkt_type = 0>" in text


def test_the_capture_is_never_waited_on_and_no_wait_is_on_a_diverted_channel():
    for d in g.designs().values():
        text = g.design_mlir(d)
        # the capture BD asks for no token, and no wait names it
        assert re.search(r"issue_token = false, metadata = @ctrl0", text)
        assert "dma_wait {symbol = @ctrl0}" not in text
        assert set(re.findall(r"dma_wait \{symbol = @(\w+)\}", text)) == {"filler0"}
        awaited = re.findall(r"aiex\.dma_await_task\(%(\w+)\)", text)
        assert awaited and all(a.startswith("mk") for a in awaited)
        for m in re.finditer(r"%(p\d+_(?:rx|tx)) = aiex\.dma_configure_task[^}]*\}[^\n]*", text):
            assert "issue_token" not in m.group(0)


def test_the_capture_bd_is_several_times_larger_than_any_word_count_the_designs_produce():
    for d in g.designs().values():
        assert d.capture_words >= 4 * sum(d.expected().values())
        assert d.capture_words * 4 % 64 == 0
        assert g.FILLER_WORDS >= 64           # a whole 256-byte burst behind the last token, whichever burst size


def test_the_filler_comes_after_the_last_marker_and_is_the_last_thing_in_the_sequence():
    text = g.design_mlir(g.designs()["cmp"])
    assert text.index("// filler") > text.index("// marker 3")
    # after the filler: one more shim loop, awaited the normal way, so its bursts reach host memory
    tail = text[text.index("// filler"):]
    assert tail.count("// marker") == 1 and tail.rstrip().splitlines()[-4].strip().startswith("aiex.dma_await_task")
    assert text.index("metadata = @ctrl0") < text.index("// probe 0")


def test_a_suppressed_probe_pushes_no_token_and_an_issuing_probe_pushes_its_id_then_bit_31_with_the_bd_id():
    d = g.designs()["cmp-half"]
    text = g.design_mlir(d)
    assert text.count("aiex.npu.write32") == len(d.probes) - len(d.suppressed)
    assert text.count("aiex.npu.maskwrite32") == len(d.probes) - len(d.suppressed)
    full = g.designs()["cmp"]
    ftext = g.design_mlir(full)
    for k, p in enumerate(full.probes):
        assert f"arith.constant {p.token_id << 8} : i32" in ftext
        rx, tx = g.probe_bd_ids(full, k)
        assert f"arith.constant {g.ENABLE_TOKEN_ISSUE | (rx if p.direction == 's2mm' else tx)} : i32" in ftext
    assert f"arith.constant {g.CONTROLLER_ID_MASK} : i32" in ftext


def test_channel_registers_follow_the_row_and_the_am025_names():
    d = g.designs()
    ctrl, queue = g.channel_registers(d["cmp"].probes[2])        # compute tile (0, 4) MM2S 0
    assert ctrl >> 20 == 4 and queue == ctrl + 4 and ctrl & 0xFFFFF == 0x1DE10
    shim_q = g.channel_registers(d["shm"].probes[0])[1]
    assert shim_q & 0xFFFFF == 0x1D204 and shim_q >> 20 == 0     # the shim names it Task_Queue
    mem = [q for q in d["memm"].probes if q.index == 5][0]
    assert g.channel_registers(mem)[0] >> 20 == 1


def test_bd_ids_are_never_reused_on_a_tile_and_memtile_odd_channels_use_the_upper_half():
    for d in g.designs().values():
        used: dict[str, set[int]] = {}
        for k, p in enumerate(d.probes):
            rx, tx = g.probe_bd_ids(d, k)
            ids = used.setdefault(p.tile, set())
            assert not ({rx, tx} & ids), (d.name, k)
            ids |= {rx, tx}
            if p.kind == "mem":
                assert (rx >= 24) == (p.index % 2 == 1) and (tx >= 24) == (p.index % 2 == 1) and tx < 48
            else:
                assert max(rx, tx) < 16
        assert not (used.get("%t00", set()) & {g.MARKER_BD_IN, g.MARKER_BD_OUT, g.CAPTURE_BD, g.FILLER_BD})


def test_the_memtile_is_measured_ascending_and_descending_per_direction():
    d = g.designs()
    for name, direction in (("mems", "s2mm"), ("memm", "mm2s")):
        up, down = d[name].probes, d[name + "r"].probes
        assert [p.index for p in up] == list(range(6)) and [p.index for p in down] == list(reversed(range(6)))
        assert {p.direction for p in up + down} == {direction}
        assert len(g.design_mlir(d[name]).split("aie.packet_flow(")) - 1 == 6 + 1     # 6 tokens + filler


def test_the_designs_have_no_cores_and_no_locks():
    for d in g.designs().values():
        text = g.design_mlir(d)
        assert "aie.core" not in text and "use_lock" not in text and "aie.lock" not in text


def test_the_shim_design_needs_two_columns_and_routes_the_second_shim_to_the_first():
    d = g.designs()["shm"]
    assert d.device == "npu1_2col" and d.columns == 2
    text = g.design_mlir(d)
    assert "aie.packet_source<%t10, TileControl : 0>" in text
    assert f"pkt_id = {g.SHIM1_CONTROLLER_ID}" in text


def test_host_input_carries_the_recognisable_filler_in_its_last_words():
    d = g.designs()["shm"]
    raw = g.host_in(d)
    words = [int.from_bytes(raw[4 * i:4 * i + 4], "little") for i in range(d.host_in_words)]
    assert words[d.filler_offset:] == [g.FILLER_MARK + j for j in range(g.FILLER_WORDS)]
    assert words[0] == g.HOST_MARK


# -- decoding ------------------------------------------------------------------------------

def _word(col, row, actor, ident, typ=6):
    w = col << 21 | row << 16 | typ << 12 | actor << 8 | ident
    return w | ((bin(w).count("1") % 2 == 0) << 31)


def _filler():
    return [0x80200000 | g.FILLER_ID] + [g.FILLER_MARK + j for j in range(g.FILLER_WORDS - 2)]


def _capture(d, tokens=None):
    """What the capture holds: the token words, the filler (its tail cut), zeros after."""
    toks = tokens if tokens is not None else [_word(1, p.row, k, p.token_id) for k, p in enumerate(d.probes)
                                              if k not in d.suppressed]
    words = toks + _filler()
    return words + [0] * (d.capture_words - len(words))


def test_a_complete_capture_is_valid_and_every_token_is_attributed_to_its_probe():
    d = g.designs()["cmp"]
    dec = g.decode_capture(_capture(d), d)
    assert dec.valid and dec.token_words == 4 and not dec.notes
    assert [r.probe for r in dec.records if r.kind == "token"] == list(d.probes)
    assert dec.filler_at == 4


def test_the_decoder_does_not_assume_one_word_per_token():
    d = g.designs()["cmp"]
    toks = [_word(1, 2, 0, 8), _word(1, 2, 0, 8), _word(1, 3, 1, 9), _word(1, 4, 6, 10), _word(1, 5, 7, 11)]
    dec = g.decode_capture(_capture(d, toks), d)
    assert dec.valid and dec.token_words == 5 and dec.found[8] == 2
    assert any(r.kind == "extra" for r in dec.records) and dec.notes == ("id 8: 2 word(s), expected 1",)


def test_a_missing_token_marks_the_run_invalid_and_names_the_probe_and_the_count_found():
    d = g.designs()["cmp"]
    toks = [_word(1, p.row, k, p.token_id) for k, p in enumerate(d.probes)][:3]
    dec = g.decode_capture(_capture(d, toks), d)
    assert not dec.valid and dec.missing == (3,) and dec.token_words == 3
    assert any("core0_5_mm2s1" in p for p in dec.problems)
    assert "INVALID" in g.report(dec, d) and "token words found 3" in g.report(dec, d)


def test_a_word_with_an_unknown_id_cannot_be_attributed():
    d = g.designs()["cmp"]
    dec = g.decode_capture(_capture(d, [_word(1, 2, 0, 8), 0x12345 << 5 | 31]), d)
    assert not dec.valid and any(r.kind == "unattributed" for r in dec.records)


def test_a_capture_without_the_filler_is_invalid_because_tokens_may_be_unflushed():
    d = g.designs()["cmp"]
    words = [_word(1, p.row, 0, p.token_id) for p in d.probes] + [0] * 20
    dec = g.decode_capture(words, d)
    assert dec.filler_at is None and not dec.valid and dec.token_words == 4


def test_the_half_suppressed_variant_expects_only_the_issuing_probes_and_reports_the_right_count():
    d = g.designs()["mems-half"]
    dec = g.decode_capture(_capture(d), d)
    assert dec.valid and dec.token_words == 3 and dec.missing == ()


def test_the_doubled_variant_expects_two_words_per_id():
    d = g.designs()["memm-dbl"]
    toks = [_word(1, 1, 16, p.token_id) for p in d.probes]
    dec = g.decode_capture(_capture(d, toks), d)
    assert dec.valid and dec.token_words == 12 and not dec.notes
    one_short = g.decode_capture(_capture(d, toks[:-1]), d)
    assert not one_short.valid and one_short.missing == (11,)


def test_tokens_are_attributed_by_id_whatever_order_they_arrive_in():
    d = g.designs()["memsr"]
    toks = [_word(1, 1, k + 1, p.token_id) for k, p in enumerate(d.probes)]
    dec = g.decode_capture(_capture(d, toks), d)
    assert dec.valid and [r.seq for r in dec.records if r.kind == "token"] == list(range(6))


def test_the_filler_is_found_with_or_without_its_header_word():
    d = g.designs()["cmp"]
    assert g.find_filler([5, 6, 0x80200002, g.FILLER_MARK, 0]) == 2
    assert g.find_filler([5, 6, g.FILLER_MARK + 3, 0]) == 2
    assert g.find_filler([5, 6, 0]) is None


def test_the_emulator_layout_decode_reads_the_fields_back():
    f = g.emu_fields(_word(1, 4, 6, 10))
    assert (f["col"], f["row"], f["type_hi"], f["actor"], f["id"]) == (1, 4, 3, 6, 10)
    assert g.emu_fields(_word(1, 1, 22, 19))["actor"] == 22
    assert g.parity_odd(_word(1, 4, 6, 10))


def test_finish_check_reports_a_task_that_started_more_often_than_it_finished():
    log = ("DMA tile(1,2) ch0 BD0 start cy=1\nDMA tile(1,2) ch0 BD0 FINISH cy=2\n"
           "DMA tile(1,3) ch1 BD4 start cy=3\nXDNA_EMU_STATUS: halt_reason=completed cycles=5\n")
    unfinished, completed = g.finish_check(log)
    assert completed and unfinished == ["tile(1,3) ch1 BD4: 1 started, 0 finished"]
    assert g.finish_check("DMA tile(1,0) ch0 BD15 start\nXDNA_EMU_STATUS: halt_reason=completed\n") == ([], True)
    assert g.finish_check("DMA tile(1,2) ch0 BD0 start\nDMA tile(1,2) ch0 BD0 FINISH\n") == ([], False)


def test_a_doubled_run_read_against_the_base_design_reports_the_extra_words_as_a_note_not_an_error():
    base = g.designs()["memm"]
    dbl = g.designs()["memm-dbl"]
    toks = [_word(1, 1, 16, p.token_id) for p in dbl.probes]
    dec = g.decode_capture(_capture(base, toks), base)
    assert dec.valid and dec.token_words == 12 and len(dec.notes) == 6
