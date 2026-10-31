"""Tests for gen_mac_sparse.py and analyze_mac_sparse.py: designed inputs, layout, staged data
contract, and the silicon goldens against the reference model of the sparse rule."""

import os

import analyze_mac_sparse as analyze
import gen_mac_sparse as gen
import pytest

L1_BUDGET_WORDS = 12 * 1024    # in + out chunk buffers, well inside the 64 KiB data memory


def golden_exists(mode, kind):
    return os.path.exists(gen.golden_path(mode, kind))


class TestDesign:
    def test_masks_set_every_bit_and_include_the_extremes(self):
        masks = gen.mask_patterns()
        assert 0 in masks and (1 << gen.MASK_BITS) - 1 in masks
        union = 0
        for m in masks:
            union |= m
        assert union == (1 << gen.MASK_BITS) - 1
        assert len(masks) == len(set(masks))

    def test_every_single_mask_bit_is_exercised_alone_or_in_a_sparse_mask(self):
        covered = set()
        for m in gen.mask_patterns():
            if bin(m).count("1") <= 4:
                covered |= {b for b in range(gen.MASK_BITS) if m >> b & 1}
        assert covered == set(range(gen.MASK_BITS))

    @pytest.mark.parametrize("mode", gen.MODES, ids=lambda m: m.stem)
    @pytest.mark.parametrize("kind", list(gen.KINDS))
    def test_tiles_fill_whole_chunks_inside_the_budget(self, mode, kind):
        tiles = gen.tiles_of(mode, kind)
        t = gen.tiles_per_chunk(mode, kind)
        assert len(tiles) % t == 0 and len(tiles) // t <= gen.MAX_CHUNKS
        assert t * gen.words_per_tile_in(mode, kind) + t * gen.words_per_tile_out(mode, kind) <= L1_BUDGET_WORDS
        for tile in tiles:
            assert len(tile) == gen.KINDS[kind].sets
            for s in tile:
                assert len(s.a) == gen.a_lanes_of(mode, kind)
                assert len(s.d) == mode.b_stored
                assert 0 <= s.m < 1 << gen.MASK_BITS

    @pytest.mark.parametrize("mode", gen.MODES, ids=lambda m: m.stem)
    def test_input_words_match_the_declared_chunk_size(self, mode):
        for kind in gen.KINDS:
            tiles = gen.tiles_of(mode, kind)
            words = gen.pack_input(mode, kind, tiles)
            chunks = len(tiles) // gen.tiles_per_chunk(mode, kind)
            assert len(words) == chunks * gen.tiles_per_chunk(mode, kind) * gen.words_per_tile_in(mode, kind)

    def test_position_tiles_use_distinct_powers_of_two_per_lane_sum(self):
        for mode in gen.MODES:
            rows, inner, _ = mode.tile
            a = gen.position_a(mode)
            for r in range(rows):
                used = [a[r * inner + k] for k in range(inner) if a[r * inner + k]]
                assert len(used) == len(set(used))

    def test_identity_tiles_cover_every_k_once(self):
        for mode in gen.MODES:
            rows, inner, _ = mode.tile
            seen = []
            for grp in range(inner // rows):
                a = gen.identity_a(mode, grp)
                seen += [k for r in range(rows) for k in range(inner) if a[r * inner + k]]
            assert sorted(seen) == list(range(inner))

    def test_four_bit_identity_ramps_name_all_stored_indices(self):
        mode = gen.BY_STEM["4x32_32x8"]
        lo, hi = gen.identity_ramps(mode)
        assert [a + 16 * b for a, b in zip(lo, hi)] == list(range(mode.b_stored))

    def test_held_out_inputs_are_not_the_probe_masks(self):
        mode = gen.BY_STEM["4x16_16x8"]
        probe = set(gen.mask_patterns())
        rand = {s.m for tile in gen.tiles_of(mode, "rand") for s in tile}
        assert len(rand - probe) > 0.9 * len(rand)

    def test_kernel_names_are_unique(self):
        names = gen.kernel_names()
        assert len(names) == len(set(names)) == len(gen.MODES) * len(gen.KINDS)

    def test_wide_modes_are_the_ones_with_a_1024_bit_operand(self):
        assert {m.stem for m in gen.MODES if m.wide} == {"4x32_32x8", "4x16_16x4"}


class TestRule:
    """The sparse rule on hand-made cases (the silicon goldens test it on real data)."""

    def mode(self):
        return gen.BY_STEM["4x16_16x8"]

    def test_a_single_bit_moves_stored_element_zero_to_its_logical_position(self):
        mode = self.mode()
        data = list(range(1, 65))
        # bit 17: element n = 17 -> column 1, k = 1, group 0, lone element at position 1 -> u = 0
        b = analyze.expand_b(mode, data, 1 << 17)
        assert b[(1, 1)] == data[0 * 8 + 1]
        assert sum(1 for v in b.values() if v) == 1

    def test_a_lone_element_in_the_upper_half_of_a_group_takes_the_second_slot(self):
        mode = self.mode()
        data = list(range(1, 65))
        b = analyze.expand_b(mode, data, 1 << 2)    # column 0, k = 2
        assert b[(2, 0)] == data[1 * 8 + 0]

    def test_two_elements_take_the_two_slots_in_ascending_order(self):
        mode = self.mode()
        data = list(range(1, 65))
        b = analyze.expand_b(mode, data, (1 << 1) | (1 << 2))
        assert b[(1, 0)] == data[0] and b[(2, 0)] == data[8]

    def test_a_group_with_three_elements_contributes_nothing(self):
        mode = self.mode()
        data = list(range(1, 65))
        assert not any(analyze.expand_b(mode, data, 0b0111).values())

    def test_sixteen_bit_elements_are_selected_by_either_of_their_two_bits(self):
        mode = gen.BY_STEM["2x8_8x8"]
        data = list(range(1, 33))
        assert analyze.expand_b(mode, data, 1 << 1)[(0, 0)] == data[0]
        assert analyze.expand_b(mode, data, 1 << 0)[(0, 0)] == data[0]

    def test_four_bit_mask_bits_select_a_byte_of_two_neighbouring_columns(self):
        mode = gen.BY_STEM["4x32_32x8"]
        data = list(range(1, 17)) * 8
        b = analyze.expand_b(mode, data, 1 << 0)
        assert b[(0, 0)] == data[0] and b[(0, 1)] == data[1]
        assert sum(1 for v in b.values() if v) == 2


@pytest.mark.parametrize("mode", gen.MODES, ids=lambda m: m.stem)
@pytest.mark.parametrize("kind", list(gen.KINDS))
def test_silicon_goldens_match_the_model(mode, kind):
    if not golden_exists(mode, kind):
        pytest.skip("not captured")
    gold = analyze.golden(gen.kernel_name(mode, kind))
    checked, bad, first = analyze.check_capture(mode, gold)
    assert checked > 0 and bad == 0, first
