"""R3b-PC control-packet request-binary tests (#140 SP-5b): byte layout, parity,
opcode, and target address of the OP_READ request stream, against a frozen fixture
and the control-header formula in AIETargetNPU.cpp:331-336. One control-header word
per tile (routing is via the kernel's `packet` attribute, not an in-band word)."""
import struct
from calibration.skew.r3b_ctrlpkt import build_ctrlpkt, _parity, _read_header

# pkt_in/pkt_out are <= 31 (5-bit packet-id field, AIEDialect.cpp:2307). pkt_in
# is unused by the generator (kernel-side routing attribute); kept for realism.
GEOM = {
    "sources": {"s1": {"col": 0, "row": 0}, "s2": {"col": 2, "row": 5}},
    "tiles": [
        {"col": 0, "row": 3, "counter_index": 0, "pkt_in": 8, "pkt_out": 22},
        {"col": 1, "row": 3, "counter_index": 1, "pkt_in": 18, "pkt_out": 9},
    ],
}


def test_one_word_per_tile_in_counter_index_order():
    buf = build_ctrlpkt(GEOM, counter_offset=0x31520)
    assert len(buf) == len(GEOM["tiles"]) * 4  # 1 word/tile
    words = struct.unpack("<%dI" % (len(buf) // 4), buf)
    # word k is tile k's control header (counter_index order).
    assert (words[0] & 0xFFFFF) == 0x31520          # control header addr field
    assert (words[1] & 0xFFFFF) == 0x31520


def test_control_header_encodes_read_opcode_and_stream_id():
    buf = build_ctrlpkt(GEOM)
    words = struct.unpack("<%dI" % (len(buf) // 4), buf)
    assert (words[0] >> 22) & 0x3 == 1          # opcode = READ
    assert (words[0] >> 20) & 0x3 == 0          # beats = 0 (single word read)
    assert (words[0] >> 24) & 0x7F == 22        # stream_id = pkt_out of tile 0
    assert (words[1] >> 24) & 0x7F == 9         # stream_id = pkt_out of tile 1


def test_odd_parity_bit_31_on_every_word():
    # AIETargetNPU.cpp:328/336 sets bit 31 so the FULL word has ODD popcount.
    buf = build_ctrlpkt(GEOM)
    for w in struct.unpack("<%dI" % (len(buf) // 4), buf):
        assert bin(w).count("1") % 2 == 1, hex(w)   # odd parity over the full 32 bits


def test_read_header_matches_formula():
    # _read_header(stream_id, opcode, beats, addr) == parity | fields.
    h = _read_header(stream_id=22, opcode=1, beats=0, addr=0x31520)
    assert (h >> 24) & 0x7F == 22 and (h >> 22) & 0x3 == 1 and (h & 0xFFFFF) == 0x31520
    assert _parity(h) == 1                       # odd overall parity
