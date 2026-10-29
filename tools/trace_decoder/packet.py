# SPDX-License-Identifier: MIT
"""Stream-switch packet header parsing and de-interleaving.

Trace data leaves the array as 32-byte stream-switch packets.  Each
packet begins with a single 32-bit header word identifying the source
tile and packet type, followed by 7 payload words (28 bytes) of trace
opcodes.

Packets from different tiles arrive interleaved on the shared S2MM DMA
channel; de-interleaving groups payload words by ``(pkt_type, row, col)``.

The header layout below matches the public ``parse_pkt_hdr_in_stream``
in mlir-aie's ``aie.utils.trace.utils`` (Apache 2.0):

  bit  31      odd parity
  bits 30..28  reserved (must be 0)
  bits 27..21  column (7 bits)
  bits 20..16  row (5 bits)
  bits 15..14  reserved (must be 0)
  bits 13..12  packet type (0=core, 1=mem, 2=shim, 3=memtile)
  bits 11..5   reserved (must be 0)
  bits  4..0   packet id

We re-implement validation (parity + reserved-zero checks) directly so
the decoder has no runtime dependency on mlir-aie.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

PACKET_WORDS = 8  # 1 header + 7 payload words = 32 bytes


@dataclass(frozen=True)
class StreamPacketHeader:
    """Decoded fields of a 32-bit trace-stream packet header."""

    col: int
    row: int
    pkt_type: int
    pkt_id: int


def _odd_parity(word: int) -> bool:
    """True iff the 32-bit word has odd 1-bit count.

    Trace packet headers carry odd parity in bit 31; valid headers must
    XOR-fold to 1.
    """
    return ((word & 0xFFFFFFFF).bit_count() & 1) == 1


def parse_packet_header(word: int) -> StreamPacketHeader | None:
    """Parse a 32-bit packet header, returning None if it is not a valid
    trace header.

    A header is considered valid when:
      1. Bit 31 carries odd parity over the whole word.
      2. The reserved bit ranges [5..10], [19], and [28..30] are zero
         (matches mlir-aie's ``parse_pkt_hdr_in_stream``).

    Anything else is treated as payload or padding by callers.
    """
    w = int(word) & 0xFFFFFFFF
    if not _odd_parity(w):
        return None
    if ((w >> 5) & 0x7F) != 0:
        return None
    if ((w >> 19) & 0x1) != 0:
        return None
    if ((w >> 28) & 0x7) != 0:
        return None
    return StreamPacketHeader(
        col=(w >> 21) & 0x7F,
        row=(w >> 16) & 0x1F,
        pkt_type=(w >> 12) & 0x3,
        pkt_id=w & 0x1F,
    )


def deinterleave_packets(words: Iterable[int]) -> dict[tuple[int, int, int], list[int]]:
    """Group payload words by source ``(pkt_type, row, col)``.

    Walks the 32-bit word stream in 8-word strides.  At each stride
    boundary, the leading word is parsed as a packet header; the
    following 7 words become payload for that header's tile.  Strides
    whose leading word does not parse as a header are skipped (they
    typically appear in malformed or padded captures).

    Returns a dict keyed by ``(pkt_type, row, col)`` mapping to a list
    of payload words in stream order.
    """
    by_tile: dict[tuple[int, int, int], list[int]] = {}
    word_list = list(words)
    cursor = 0
    while cursor + PACKET_WORDS <= len(word_list):
        header = parse_packet_header(word_list[cursor])
        if header is None:
            cursor += 1
            continue
        key = (header.pkt_type, header.row, header.col)
        bucket = by_tile.setdefault(key, [])
        bucket.extend(word_list[cursor + 1 : cursor + PACKET_WORDS])
        cursor += PACKET_WORDS
    return by_tile


def words_to_bytes(words: Iterable[int]) -> list[int]:
    """Flatten a payload word list into a big-endian byte stream.

    Trace opcodes are byte-oriented and packed MSB-first within each
    32-bit payload word, matching the on-tile encoder.  This conversion
    is what the downstream mode decoders consume.
    """
    out: list[int] = []
    for word in words:
        w = int(word) & 0xFFFFFFFF
        out.append((w >> 24) & 0xFF)
        out.append((w >> 16) & 0xFF)
        out.append((w >> 8) & 0xFF)
        out.append(w & 0xFF)
    return out


# Sentinel byte sequences emitted by the on-tile encoder when no event
# fires for a long stretch.  Convert to bytes() once so callers can use
# them in fast comparisons.
PADDING_WORDS = (
    "fefefefe",  # idle filler
    "a5a5a5a5",  # alignment marker
)


def trim_trailing_padding(words: list[int]) -> list[int]:
    """Drop the trailing padding emitted by the encoder when the trace
    buffer is larger than the actual payload.

    The encoder caps the payload with a 0xFEFEFEFE word followed by a
    run of zero words.  We treat the first 0xFEFEFEFE that is followed
    by at least two zero words as the end-of-payload marker.
    """
    target_pad = 0xFEFEFEFE
    for i, w in enumerate(words):
        if w != target_pad:
            continue
        if i + 2 < len(words) and words[i + 1] == 0 and words[i + 2] == 0:
            return words[: i + 1]
    return words
