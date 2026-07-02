"""R3b-PC control-packet OP_READ request-binary generator (#140 SP-5b).

`aiex.npu.write32` is write-only, so R3b reads each measured tile's
Performance_Counter0 back off-chip via a hand-assembled control-packet
OP_READ stream instead. This module builds that request binary, consumed by
`bridge-trace-runner --ctrlpkt`: two little-endian u32 words per measured
tile (a routing header, then a control header), in `counter_index` order --
exactly the layout `r3b_observe.observe_r3b` parses back out of the
readback buffer.

Hardware facts this module encodes (not tool internals):
- `0x31520` is Performance_Counter0's tile-local address: AM025 register
  database entry `Performance_Counter0`, and aie-rt
  `driver/src/global/xaiemlgbl_params.h:2264`
  (`XAIEMLGBL_CORE_MODULE_PERFORMANCE_COUNTER0 == 0x00031520`).
- The 2-word-per-packet form (routing header + control header, no payload
  words for a read) matches the `aiex.npu.control_packet` lowering used by
  `mlir-aie/test/npu-xrt/add_one_ctrl_packet/aie.mlir` and implemented in
  `mlir-aie/lib/Targets/AIETargetNPU.cpp:277-344`
  (`AIETranslateControlPacketsToUI32Vec`).
- `opcode == 1` is OP_READ; see
  `mlir-aie/test/npu-xrt/debug_halt_probe/test.cpp:157`.
- Field layout, per `AIETargetNPU.cpp:319-336`:
    routing header  = (pkt_type & 0x7) << 12 | (pkt_id & 0xff)
    control header  = stream_id << 24 | opcode << 22 | beats << 20
                       | (addr & 0xFFFFF)
- Parity, per `AIETargetNPU.cpp:309-328,336`: bit 31 of each word is set to
  1 iff the low-31-bit header has an EVEN number of set bits. That forces
  the FULL 32-bit word to always have an ODD number of set bits -- i.e.
  odd parity over the whole word, not even (verified directly against the
  cited source; the `debug_halt_probe/test.cpp` comment mislabels the same
  formula "even-parity", but the code there computes the identical odd
  result). See task-2-report.md for the worked derivation.
"""
import struct


def _parity(w):
    """XOR of all 32 bits of `w` (1 if `w` has an odd number of set bits,
    0 if even). A correctly-encoded header word from this module always has
    `_parity(w) == 1` (odd parity over the full word)."""
    return bin(w & 0xFFFFFFFF).count("1") % 2


def _set_parity_bit(low31):
    """Set bit 31 of a header built from its low 31 bits so the resulting
    32-bit word has odd overall parity, matching AIETargetNPU.cpp's
    `parity(hdr)` lambda: bit31 = 1 iff popcount(low31) is even."""
    even = bin(low31 & 0x7FFFFFFF).count("1") % 2 == 0
    return (low31 & 0x7FFFFFFF) | ((1 if even else 0) << 31)


def _routing_header(pkt_id, pkt_type=0):
    """Stream-switch routing header: word[0] of a control packet."""
    hdr = ((pkt_type & 0x7) << 12) | (pkt_id & 0xFF)
    return _set_parity_bit(hdr)


def _read_header(stream_id, opcode, beats, addr):
    """Control header for an OP_READ: word[1] of a control packet."""
    hdr = (((stream_id & 0x7F) << 24) | ((opcode & 0x3) << 22)
           | ((beats & 0x3) << 20) | (addr & 0xFFFFF))
    return _set_parity_bit(hdr)


def build_ctrlpkt(geometry, counter_offset=0x31520):
    """Build the `--ctrlpkt` OP_READ request binary for every measured tile
    in `geometry` (as produced by Task 1's geometry.json). Two words per
    tile -- `[routing_header(pkt_id=tile.pkt_in),
    control_header(stream_id=tile.pkt_out, opcode=1, beats=0,
    addr=counter_offset)]` -- ordered by `counter_index` so the request
    stream lines up with the readback buffer `observe_r3b` expects.
    """
    words = []
    for tile in sorted(geometry["tiles"], key=lambda t: t["counter_index"]):
        words.append(_routing_header(pkt_id=tile["pkt_in"]))
        words.append(_read_header(stream_id=tile["pkt_out"], opcode=1,
                                   beats=0, addr=counter_offset))
    return struct.pack("<%dI" % len(words), *words)
