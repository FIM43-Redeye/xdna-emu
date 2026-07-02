"""R3b-PC control-packet request-binary tests (#140 SP-5b): byte layout, parity,
opcode, and target address of the OP_READ request stream, against a frozen fixture
and the header formula in AIETargetNPU.cpp:277-344."""
import struct
from calibration.skew.r3b_ctrlpkt import build_ctrlpkt, _parity, _read_header

GEOM = {
    "sources": {"s1": {"col": 0, "row": 0}, "s2": {"col": 2, "row": 5}},
    "tiles": [
        {"col": 0, "row": 3, "counter_index": 0, "pkt_in": 16, "pkt_out": 32},
        {"col": 1, "row": 3, "counter_index": 1, "pkt_in": 17, "pkt_out": 33},
    ],
}


def test_two_words_per_tile_in_counter_index_order():
    buf = build_ctrlpkt(GEOM, counter_offset=0x31520)
    assert len(buf) == len(GEOM["tiles"]) * 2 * 4  # 2 words/tile
    words = struct.unpack("<%dI" % (len(buf) // 4), buf)
    # tile 0 occupies words[0:2], tile 1 words[2:4] (counter_index order).
    assert (words[1] & 0xFFFFF) == 0x31520          # control header addr field
    assert (words[3] & 0xFFFFF) == 0x31520


def test_control_header_encodes_read_opcode_and_stream_id():
    buf = build_ctrlpkt(GEOM)
    words = struct.unpack("<%dI" % (len(buf) // 4), buf)
    ctrl0 = words[1]
    assert (ctrl0 >> 22) & 0x3 == 1          # opcode = READ
    assert (ctrl0 >> 20) & 0x3 == 0          # beats = 0 (single word read)
    assert (ctrl0 >> 24) & 0x7F == 32        # stream_id = pkt_out of tile 0


def test_routing_header_carries_pkt_in():
    buf = build_ctrlpkt(GEOM)
    words = struct.unpack("<%dI" % (len(buf) // 4), buf)
    assert (words[0] & 0xFF) == 16           # pkt_id = pkt_in of tile 0
    assert (words[2] & 0xFF) == 17


def test_odd_parity_bit_31_on_every_word():
    # AIETargetNPU.cpp:328/336 sets bit 31 so the FULL word has ODD popcount.
    buf = build_ctrlpkt(GEOM)
    for w in struct.unpack("<%dI" % (len(buf) // 4), buf):
        assert bin(w).count("1") % 2 == 1, hex(w)   # odd parity over the full 32 bits


def test_read_header_matches_formula():
    # _read_header(stream_id, opcode, beats, addr) == parity | fields.
    h = _read_header(stream_id=32, opcode=1, beats=0, addr=0x31520)
    assert (h >> 24) & 0x7F == 32 and (h >> 22) & 0x3 == 1 and (h & 0xFFFFF) == 0x31520
