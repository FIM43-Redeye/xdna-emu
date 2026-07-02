"""R3b observation-bridge tests (#140 SP-5b, rev3): readback buffer + geometry
-> design-row coefficients, against a frozen fixture."""
import struct
import pytest
from calibration.skew.r3b_observe import observe_r3b

# s1 at (0,0), s2 at (2,4) [opposite corner of a 3-col x 5-row virtual frame].
GEOM = {
    "sources": {"s1": {"col": 0, "row": 0}, "s2": {"col": 2, "row": 4}},
    "tiles": [
        {"col": 1, "row": 2, "counter_index": 0},  # interior
        {"col": 2, "row": 2, "counter_index": 1},  # east edge
        {"col": 0, "row": 2, "counter_index": 2},  # west edge
    ],
}


def _buf(values):
    return b"".join(struct.pack("<I", v) for v in values)


def test_coefficients_match_hand_computation():
    obs = observe_r3b(_buf([1000, 1010, 1020]), GEOM)
    # tile (1,2): hops(s1)= (e1,w1,n1,s1)=(1,0,2,0); hops(s2)=(e2,w2,n2,s2)=(0,1,0,2)
    #   a_hE = 0-1 = -1, a_hW = 1-0 = 1, a_vN = 0-2 = -2, a_vS = 2-0 = 2
    #   a_turn = (0+1)*(0+2) - (1+0)*(2+0) = 2 - 2 = 0
    o0 = obs[0]
    assert (o0["a_hE"], o0["a_hW"], o0["a_vN"], o0["a_vS"], o0["a_turn"]) == (-1, 1, -2, 2, 0)
    assert o0["r"] == 1000.0


def test_short_buffer_fails_loud():
    with pytest.raises(ValueError):
        observe_r3b(_buf([1000, 1010]), GEOM)  # 3 tiles, 2 words


def test_reads_counter_by_index():
    obs = observe_r3b(_buf([7, 8, 9]), GEOM)
    assert [o["r"] for o in obs] == [7.0, 8.0, 9.0]
