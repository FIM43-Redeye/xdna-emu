"""R3b-PC control-packet OP_READ request-binary generator (#140 SP-5b).

`aiex.npu.write32` is write-only, so R3b reads each measured tile's
Performance_Counter0 back off-chip via a hand-assembled control-packet
OP_READ stream instead. This module builds that request binary, consumed by
`bridge-trace-runner --ctrlpkt`: ONE little-endian u32 word per measured tile
-- the OP_READ control header -- in `counter_index` order.

The kernel routes each request to its tile via a per-push `packet` attribute
(`packet = <pkt_id = tile.pkt_in, pkt_type = 1>` on a packet-mode
`dma_memcpy_nd`, the add_one_ctrl_packet_4_cores mechanism), NOT via an in-band
routing word: the shim control-packet DMA applies its own stream header, so an
in-band routing word would arrive at the tile as payload and be mis-parsed as a
control opcode (this wedged the emulator during Task-3 bring-up). Hence the
request payload is a bare control header, and only `pkt_out` (the response
stream id the control header carries) is needed here; `pkt_in` lives in the
kernel's `packet` attribute and `packet_flow`s. The kernel's k-th push reads
word k of this binary.

Hardware facts this module encodes (not tool internals):
- `0x31520` is Performance_Counter0's tile-local address: AM025 register
  database entry `Performance_Counter0`, and aie-rt
  `driver/src/global/xaiemlgbl_params.h:2264`
  (`XAIEMLGBL_CORE_MODULE_PERFORMANCE_COUNTER0 == 0x00031520`).
- `opcode == 1` is OP_READ; see
  `mlir-aie/test/npu-xrt/debug_halt_probe/test.cpp:157`.
- Control-header field layout, per `AIETargetNPU.cpp:331-336`:
    control header = stream_id << 24 | opcode << 22 | beats << 20
                     | (addr & 0xFFFFF)
- Parity, per `AIETargetNPU.cpp:309-336`: bit 31 of each word is set to
  1 iff the low-31-bit header has an EVEN number of set bits. That forces
  the FULL 32-bit word to always have an ODD number of set bits -- i.e.
  odd parity over the whole word, not even (verified directly against the
  cited source; the `debug_halt_probe/test.cpp` comment mislabels the same
  formula "even-parity", but the code there computes the identical odd
  result).
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


def _read_header(stream_id, opcode, beats, addr):
    """Control header for an OP_READ: the single word pushed per tile."""
    hdr = (((stream_id & 0x7F) << 24) | ((opcode & 0x3) << 22)
           | ((beats & 0x3) << 20) | (addr & 0xFFFFF))
    return _set_parity_bit(hdr)


def build_ctrlpkt(geometry, counter_offset=0x31520):
    """Build the `--ctrlpkt` OP_READ request binary for every measured tile in
    `geometry` (as produced by Task 1's geometry.json). ONE word per tile -- the
    OP_READ control header `[stream_id=tile.pkt_out, opcode=1, beats=0,
    addr=counter_offset]` -- ordered by `counter_index`, so the kernel's k-th
    packet-mode push (routed to its tile by its own `packet` attribute) reads
    word k. `pkt_in` is deliberately not used here: request routing is carried
    by the kernel's `packet` attribute, not by an in-band word.
    """
    words = [
        _read_header(stream_id=tile["pkt_out"], opcode=1, beats=0,
                     addr=counter_offset)
        for tile in sorted(geometry["tiles"], key=lambda t: t["counter_index"])
    ]
    return struct.pack("<%dI" % len(words), *words)
