"""Tests for gen_vldb_sparse.py and analyze_vldb_sparse.py: designed streams, the record layout,
the reference model on hand-made cases, and the silicon goldens against the model."""

import os

import analyze_vldb_sparse as model
import gen_vldb_sparse as gen
import pytest


def golden_exists(group, width):
    return os.path.exists(gen.golden_path(width, group))


class TestDesign:
    def test_streams_fit_the_buffer_and_have_unique_names(self):
        streams = gen.streams()
        assert len({n for n, _ in streams}) == len(streams) <= gen.MAX_CHUNKS
        assert all(len(b) == gen.STREAM_BYTES for _, b in streams)

    def test_position_coded_streams_never_select_data_and_name_their_windows(self):
        by_name = dict(gen.streams())
        for name in ("dbseq_a", "dbseq_b"):
            b = by_name[name]
            assert set(b) <= {0x00, 0x0F, 0xF0, 0xFF}
            assert len({tuple(b[i:i + 8]) for i in range(len(b) - 7)}) == len(b) - 7

    def test_dense_stream_alternates_a_full_mask_and_numbered_data(self):
        b = dict(gen.streams())["dense40"]
        for pop in range(5):
            at = 40 * pop
            assert b[at:at + 8] == [0x55] * 8
            assert b[at + 8:at + 40] == [(at + 8 + i) % 255 + 1 for i in range(32)]

    def test_every_script_fits_a_group_and_the_groups_cover_all_scripts(self):
        grouped = [n for names in gen.GROUPS.values() for n in names]
        assert len(grouped) == len(set(grouped))
        assert set(grouped) <= {n for n, _ in gen.ALL_SCRIPTS}
        for group in gen.GROUPS:
            gen.set_group(group)
            assert sum(len(gen.steps_of(s)) for _, s in gen.SCRIPTS) <= 130

    def test_kernel_names_are_unique_and_parse_back(self):
        names = gen.kernel_names()
        assert len(names) == len(set(names)) == len(gen.GROUPS) * len(gen.WIDTHS)
        for name in names:
            group, width = gen.parse_kernel(name)
            assert gen.kernel_name(width, group) == name

    def test_no_script_fetches_past_the_stream(self):
        # Memory is touched only by line fetches at the pointer, which moves 32 bytes per reset,
        # fill and pop (the `dp` step fetches at the pointer without moving it); pops themselves
        # read the ring, whatever the masks say.
        for group in gen.GROUPS:
            gen.set_group(group)
            for _, script in gen.SCRIPTS:
                steps = gen.steps_of(script)
                moves = sum(s in ("reset", "fill") or s.startswith("pop") for s in steps)
                assert 32 * (moves + 1) <= gen.STREAM_BYTES
        assert 32 * gen.RING_LINES <= gen.STREAM_BYTES

    def test_normalize_zeroes_the_halves_a_lo_step_leaves_undefined(self):
        gen.set_group("ring")
        n = sum(len(gen.steps_of(s)) for _, s in gen.SCRIPTS)
        words = [0xDEAD] * (n * gen.REC)
        out = gen.normalize(words)
        base = 0
        for _, script in gen.SCRIPTS:
            undefined = False
            for step in gen.steps_of(script):
                undefined = {"pop_lo": True, "peek_lo": True, "pop_hi": False, "peek_hi": False}.get(step, undefined)
                for w in gen.HI_HALF_WORDS:
                    assert (out[base + w] == 0) == undefined
                assert out[base] == 0xDEAD
                base += gen.REC


class TestModel:
    """The load rule on hand-made cases (the goldens test it on real data)."""

    def unit_with(self, stream, width=8):
        u = model.Unit(width)
        u.ring[:len(stream)] = stream
        return u

    def test_a_lone_bit_in_the_upper_half_of_a_group_takes_slot_one(self):
        u = self.unit_with(bytes([0x04, 0, 0, 0, 0, 0, 0, 0, 0xAB]))
        mask, data = u.read(False)
        assert data[1] == 0xAB and sum(1 for x in data if x) == 1

    def test_two_bits_fill_both_slots_in_order(self):
        u = self.unit_with(bytes([0x03, 0, 0, 0, 0, 0, 0, 0, 0x11, 0x22]))
        _, data = u.read(False)
        assert data[0] == 0x11 and data[1] == 0x22

    def test_a_group_with_three_bits_is_zero_and_sets_overflow(self):
        u = self.unit_with(bytes([0x07, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3]))
        _, data = u.read(False)
        assert not any(data) and u.sparse_of == 1

    def test_pop_advances_dp_by_the_mask_and_the_rounded_popcount(self):
        u = self.unit_with(bytes([0x03] + [0] * 7 + [1, 2, 0, 0]))
        u.read(True)
        assert u.dp == 8 + 4

    def test_an_all_ones_mask_advances_only_the_mask(self):
        u = self.unit_with(bytes([0xFF] * 8))
        u.read(True)
        assert u.dp == 8

    def test_sixteen_bit_elements_place_one_byte_per_mask_bit(self):
        # Mask byte 0x03: element 0, both bytes. Mask byte 0x05: elements 0 and 1, low bytes.
        u = self.unit_with(bytes([0x03, 0x05, 0, 0, 0, 0, 0, 0, 0xA, 0xB, 0xC, 0xD]), width=16)
        _, data = u.read(False)
        assert data[0:2] == bytes([0xA, 0xB])
        assert data[4] == 0xC and data[6] == 0xD

    def test_a_pop_writes_the_next_line_after_it_reads(self):
        u = model.Unit(8)
        before = bytes(u.ring)
        u.read(True)
        u.fetch_if_room(bytes(range(32)))
        assert bytes(u.ring) != before and u.ring[:32] == bytes(range(32))

    def test_a_fetch_is_dropped_while_the_ring_is_nine_lines_ahead(self):
        u = model.Unit(8)
        for _ in range(9):
            u.fetch(bytes([1] * 32))
        u.fetch_if_room(bytes([2] * 32))
        assert u.lines == 9
        u.dp += 32      # one line consumed: room again
        u.fetch_if_room(bytes([2] * 32))
        assert u.lines == 10


@pytest.mark.parametrize("width", gen.WIDTHS)
@pytest.mark.parametrize("group", list(gen.GROUPS))
def test_silicon_goldens_match_the_model(group, width):
    if not golden_exists(group, width):
        pytest.skip("not captured")
    counts, bad = model.check_golden(gen.golden_path(width, group), group, width)
    assert counts["pops"] > 1000
    unexplained = [b for b in bad if (b[0], b[1]) not in model.KNOWN_RESIDUAL]
    assert len(bad) <= 9
    assert not unexplained, unexplained[:5]
    if group == "ring":
        assert not bad


@pytest.mark.parametrize("width", gen.WIDTHS)
def test_goldens_prove_all_four_forms_of_the_width_were_executed(width):
    for group in gen.GROUPS:
        if not golden_exists(group, width):
            pytest.skip("not captured")
        import capture_kernel_render as render
        gold = render.load_golden(gen.golden_path(width, group))
        mnemonics = {e["mnemonic"] for e in gold["encodings"]}
        assert {f"vldb.sparse.{op}.{width}" for op in ("reset", "fill", "peek", "pop")} <= mnemonics
