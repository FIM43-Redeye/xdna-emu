"""The MAC conf probe generator: conf word layout, kernel text, tile layout, plan sizes."""
import random
import struct

import gen_mac_conf_probe as g


def test_conf_word_places_every_field_at_its_vecconf_bit():
    assert g.conf_word(zero=1, amode=2, bmode=3, cmode=5, sgn_y=1, sgn_x=1, shift16=1, sub_mul=1, sub_acc1=1,
                       sub_acc2=1, reserved1=3, term_neg=0xA5, reserved2=0xFF) == (
        1 | 2 << 1 | 3 << 3 | 5 << 5 | 1 << 8 | 1 << 9 | 1 << 10 | 1 << 11 | 1 << 12 | 1 << 13 | 3 << 14
        | 0xA5 << 16 | 0xFF << 24)
    assert g.conf_word(sgn_y=0, sgn_x=0) == 0


def test_tile_layout_is_operands_then_conf():
    rng = random.Random(1)
    t = g.tile_bytes(rng, 0x12345678)
    assert len(t) == g.TILE_IN
    assert struct.unpack("<I", t[384:388])[0] == 0x12345678


def test_every_kind_has_a_kernel_with_the_builtin_and_its_operands():
    for kind, (builtin, acc1, acc2) in g.KINDS.items():
        src = g.kernel_source(kind, 4)
        assert f"__builtin_aiev2_{builtin}(" in src
        assert ("c1" in src.split("r = ")[1]) == acc1 and (", c2" in src.split("r = ")[1]) == acc2


def test_valid_codes_are_distinct_and_in_range():
    codes = g.valid_codes()
    assert len(codes) == len(set(codes)) and all(0 <= c < 128 for c in codes)


def test_plans_split_into_cases_of_at_most_the_tile_budget():
    for plan in ("flags", "upper", "termneg", "random"):
        for k in g.PLANS[plan](random.Random(1)):
            assert k.in_size <= g.TILES_PER_CASE * g.TILE_IN and k.out_size <= g.TILES_PER_CASE * g.TILE_OUT
